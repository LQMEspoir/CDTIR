# File: src/tourism_ie/modeling.py
# Module responsibility: common model-loading layer: dtype, tokenizer, Transformers base model, LoRA target-module detection, parameter counts and device.
# Main data flow: model id/load options -> tokenizer/model; or model -> LoRA targets / parameter stats / device.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
import torch
from .models import resolve_model


def choose_dtype():
    """- Responsibility: selects torch dtype from device capability and user settings, balancing precision, compatibility and memory.
    - Parameters: no explicit business params (possibly self/cls).
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# Baichuan2-7B-Chat's official tokenizer_config.json has no chat_template; here we add one following the repo's own
# generation_utils.build_chat_input convention: system content is prepended verbatim (unmarked), user starts with
# <reserved_106> (id 195), assistant starts with <reserved_107> (id 196), and the generation prompt is <reserved_107>.
_BAICHUAN2_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}"
    "{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}{{ '<reserved_106>' + message['content'] }}"
    "{% elif message['role'] == 'assistant' %}{{ '<reserved_107>' + message['content'] }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<reserved_107>' }}{% endif %}"
)


def load_tokenizer(model_name_or_key: str):
    """- Responsibility: loads the tokenizer from a Hugging Face name or local dir and fixes pad-token and other train/generate required config.
    - Parameters: model_name_or_key: str.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: involves training/backprop; memory/randomness affect resource usage and reproducibility.
    """
    from transformers import AutoTokenizer
    spec=resolve_model(model_name_or_key)
    tok = AutoTokenizer.from_pretrained(spec.model_name, trust_remote_code=True, use_fast=spec.use_fast_tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if getattr(tok, "chat_template", None) is None and spec.family == "baichuan":
        tok.chat_template = _BAICHUAN2_CHAT_TEMPLATE
    # Left padding is safer for batched decoder-only generation; training collator can still pad dynamically.
    tok.padding_side = "left"
    return tok


def _patch_baichuan_legacy_cache(model) -> None:
    """- Responsibility: makes Baichuan2's legacy modeling_baichuan.py compatible with transformers>=4.40 Cache/DynamicCache.
    - Parameters: model (BaichuanForCausalLM instance).
    - Returns: return type is `None`; see implementation below for field details.
    - Notes: the legacy code receives an empty DynamicCache (not None) on the first forward, and later cache steps
            can't retrieve per-layer past_key_value, crashing `past_key_values[0][0]`/SDPA mask. Two safeguards: (1) mark the model
            stateful to force the legacy tuple cache; (2) wrap BaichuanModel.forward to normalize empty/invalid cache.
    """
    import functools
    # core fix: Baichuan2 legacy code only understands the old tuple cache; transformers>=4.40 passes DynamicCache by default,
    # per-layer past_key_value can't be retrieved -> keys aren't concatenated -> SDPA mask shape breaks. Mark stateful to force the legacy cache.
    try:
        type(model)._is_stateful = True
    except Exception:
        pass
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "forward"):
        return
    orig = inner.forward

    def _norm(pkv):
        if pkv is None:
            return None
        if hasattr(pkv, "get_seq_length") and pkv.get_seq_length() == 0:
            return None
        try:
            first = pkv[0]
        except (IndexError, KeyError, TypeError):
            return None
        if first is None or first[0] is None:
            return None
        return pkv

    @functools.wraps(orig)
    def patched(*args, **kwargs):
        kwargs["past_key_values"] = _norm(kwargs.get("past_key_values"))
        return orig(*args, **kwargs)

    inner.forward = patched


def load_base_model(model_name_or_key: str, training: bool = False, quantized: bool = False):
    """- Responsibility: loads the CausalLM base model with dtype, 4-bit/device-map options.
    - Parameters: model_name_or_key: str,training: bool=False,quantized: bool=False.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: involves training/backprop; memory/randomness affect resource usage and reproducibility;includes explicit parameter/data validation; raises on invalid input.
    """
    from transformers import AutoModelForCausalLM
    spec=resolve_model(model_name_or_key)
    dtype=choose_dtype()
    kwargs={"trust_remote_code":True,"torch_dtype":dtype,"low_cpu_mem_usage":True}
    if quantized:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA-capable environment in this project configuration.")
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise RuntimeError("QLoRA requires bitsandbytes. Install the optional dependency: pip install bitsandbytes") from e
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
        )
        kwargs["device_map"] = "auto"
    model=AutoModelForCausalLM.from_pretrained(spec.model_name, **kwargs)
    if spec.family == "baichuan":
        _patch_baichuan_legacy_cache(model)
    if training:
        model.config.use_cache=False
    return model


def find_lora_targets(model, requested: list[str] | tuple[str,...]) -> list[str]:
    """- Responsibility: scans model module names to infer LoRA-suitable linear projections while excluding output heads that should not be adapted.
    - Parameters: model,requested: list[str] | tuple[str, ...].
    - Returns: return type is `list[str]`; see implementation below for field details.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    available={name.rsplit(".",1)[-1] for name,_ in model.named_modules()}
    targets=[x for x in requested if x in available]
    if not targets:
        raise RuntimeError(
            "Could not find any supported LoRA target modules in this model. "
            f"Requested={list(requested)}. For a custom architecture, register its target modules in models/registry.py."
        )
    return targets


def count_parameters(model):
    """- Responsibility: counts total parameters and trainable (requires_grad=True) parameters for experiment resource reporting.
    - Parameters: model.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters":total,"trainable_parameters":trainable,"trainable_percent":100*trainable/max(total,1)}


def device_of_model(model):
    """- Responsibility: reliably gets the model's current device, compatible with plain models and accelerate/device_map.
    - Parameters: model.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    try: return next(model.parameters()).device
    except StopIteration: return torch.device("cpu")
