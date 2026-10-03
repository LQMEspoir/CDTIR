# File: src/tourism_ie/models/registry.py
# Module responsibility: model registry. Defines ModelSpec, and maintains the Hugging Face names, aliases, and model metadata for the Core-4 and optional models.
# Main data flow: model key / custom path -> ModelSpec; can also list the core/extended model collections.
# Reading suggestion: start with the public functions/classes in this file, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments, without modifying existing expressions, control flow, parameter defaults, or function call relationships.

from __future__ import annotations
from dataclasses import dataclass, asdict
from dataclasses import replace
import os
import re
from pathlib import Path

# Paper reproduction injects LoRA only into the four attention projections.
# relation_legacy_fixed keeps its explicit q/v tuple below.
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

@dataclass(frozen=True)
class ModelSpec:
    """[Class description] ModelSpec
    - Responsibility: data structure for model registration info, storing metadata such as the model key, actual model ID, display name, and whether it belongs to the default Core set.
    - Inheritance: does not explicitly inherit a business base class.
    - Usage: instantiated and then used by the training/inference main flow via its public methods; internal state is determined by __init__ and the saved config together.
    """
    key: str
    model_name: str
    family: str
    params: str
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    gated: bool = False
    core_benchmark: bool = True
    use_fast_tokenizer: bool = True
    notes: str = ""

    def to_dict(self):
        """[Function description] ModelSpec.to_dict
        - Responsibility: converts the dataclass model spec to a plain dict for JSON serialization and CLI display.
        - Main parameters: no explicit business parameters (may only contain self/cls).
        - Returns: returns the processed object/metrics/tensor/path, etc.; the actual structure depends on each return branch.
        - Note: this function mainly does in-memory computation; no extra persistence side effects beyond the called object's own behavior.
        """
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d

MODEL_SPECS = [
    # === Core-4 benchmark models ===
    ModelSpec("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "qwen2", "7B"),
    ModelSpec("glm4-9b", "zai-org/glm-4-9b-chat-hf", "glm4", "9B",
              use_fast_tokenizer=False,
              notes="GLM-4-9B-Chat; recent Transformers is required."),
    ModelSpec("llama3.1-8b", "meta-llama/Meta-Llama-3.1-8B-Instruct", "llama", "8B", gated=True,
              notes="Requires accepting Meta's model license on Hugging Face."),
    ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen3", "8B",
              notes="Qwen3-8B; unified 2025 backbone (thinking / non-thinking modes)."),
    # === Optional models (not in core benchmark) ===
    ModelSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "qwen2", "0.5B", core_benchmark=False),
    ModelSpec("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "qwen2", "1.5B", core_benchmark=False),
    ModelSpec("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "qwen2", "3B", core_benchmark=False),
    ModelSpec("llama3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", "llama", "1B", gated=True, core_benchmark=False,
              notes="May require accepting the model license and Hugging Face authentication."),
    ModelSpec("llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "llama", "3B", gated=True, core_benchmark=False,
              notes="May require accepting the model license and Hugging Face authentication."),
    ModelSpec("mistral7b-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "mistral", "7B", core_benchmark=False),
    ModelSpec("qwen2-1.5b-relation-legacy", "Qwen/Qwen2-1.5B-Instruct", "qwen2", "1.5B",
              target_modules=("q_proj", "v_proj"), core_benchmark=False, use_fast_tokenizer=False,
              notes="Exact backbone/LoRA target pair from legacy relation.py; used by relation_legacy_fixed."),
    # === New base models (download, run, and delete one by one) ===
    ModelSpec("deepseek-7b", "deepseek-ai/deepseek-llm-7b-chat", "deepseek", "7B", core_benchmark=False,
              notes="DeepSeek-7B-Chat; strong Chinese."),
    ModelSpec("internlm2.5-7b", "internlm/internlm2_5-7b-chat", "internlm2.5", "7B", core_benchmark=False,
              notes="InternLM2.5-7B-Chat; Shanghai AI Lab, strong Chinese, 1M context."),
    ModelSpec("baichuan2-7b", "baichuan-inc/Baichuan2-7B-Chat", "baichuan", "7B", core_benchmark=False,
              notes="Baichuan2-7B-Chat; classic Chinese backbone."),
]
MODEL_REGISTRY = {x.key: x for x in MODEL_SPECS}

_LOCAL_MODEL_NAMES = {
    "qwen2.5-0.5b": ("Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b"),
    "qwen2.5-1.5b": ("Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b"),
    "qwen2.5-3b": ("Qwen2.5-3B-Instruct", "qwen2.5-3b"),
    "qwen2.5-7b": ("Qwen2.5-7B-Instruct", "qwen2.5-7b"),
    "qwen3-8b": ("Qwen3-8B", "qwen3-8b"),
    "glm4-9b": ("glm-4-9b-chat-hf", "GLM-4-9B", "glm4-9b"),
    "llama3.1-8b": ("Meta-Llama-3.1-8B-Instruct", "LLaMA3.1-8B", "llama3.1-8b"),
    "llama3.2-1b": ("Llama-3.2-1B-Instruct", "llama3.2-1b"),
    "llama3.2-3b": ("Llama-3.2-3B-Instruct", "llama3.2-3b"),
    "mistral7b-v0.3": ("Mistral-7B-Instruct-v0.3", "mistral7b-v0.3"),
}


def _local_model_override(spec: ModelSpec) -> str | None:
    """Find a downloaded model in ``../models`` without changing CLI names.

    The repository itself stays portable: a registered key still resolves to
    its Hugging Face id when no local directory exists.  When a directory with
    config.json is present one level above the current workspace, it is used
    automatically, which keeps large model weights out of the code directory.
    ``CDTIR_MODEL_ROOT`` overrides the parent-directory convention.
    """
    names=_LOCAL_MODEL_NAMES.get(spec.key)
    if not names:
        return None
    configured=os.environ.get("CDTIR_MODEL_ROOT")
    if configured:
        model_root=Path(configured).expanduser()
    else:
        runtime_root=Path(os.environ.get("CDTIR_ROOT", Path.cwd())).expanduser().resolve()
        model_root=runtime_root.parent / "models"
    for name in names:
        candidate=model_root/name
        if (candidate/"config.json").is_file():
            return str(candidate)
    return None


def resolve_model(name_or_key: str) -> ModelSpec:
    """[Function description] resolve_model
    - Responsibility: resolves a model alias, registry key, or custom Hugging Face/local path into a unified ModelSpec.
    - Main parameters: name_or_key: str.
    - Returns: return type annotated as `ModelSpec`; see the implementation below for field meanings.
    - Note: this function mainly does in-memory computation; no extra persistence side effects beyond the called object's own behavior.
    """
    # Explicit local paths should keep the registered canonical key (so
    # benchmark summaries and paper tables still identify Qwen/GLM/LLaMA),
    # while loading from the path supplied by the caller.
    local_name=Path(name_or_key).expanduser().name
    if local_name and local_name != name_or_key:
        for spec in MODEL_SPECS:
            if local_name in _LOCAL_MODEL_NAMES.get(spec.key, ()):
                return replace(spec,model_name=name_or_key)
    if name_or_key in MODEL_REGISTRY:
        spec=MODEL_REGISTRY[name_or_key]
        local=_local_model_override(spec)
        return replace(spec,model_name=local) if local else spec
    for spec in MODEL_SPECS:
        if name_or_key == spec.model_name:
            local=_local_model_override(spec)
            return replace(spec,model_name=local) if local else spec
    base = Path(name_or_key).name or "custom_model"
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:80]
    return ModelSpec(key, name_or_key, "custom", "unknown", core_benchmark=False)


def list_models(core_only: bool = False) -> list[ModelSpec]:
    """[Function description] list_models
    - Responsibility: returns a serializable description of the current model registry for CLI display or auditing experiment configuration.
    - Main parameters: core_only: bool=False.
    - Returns: return type annotated as `list[ModelSpec]`; see the implementation below for field meanings.
    - Note: this function mainly does in-memory computation; no extra persistence side effects beyond the called object's own behavior.
    """
    return [x for x in MODEL_SPECS if x.core_benchmark] if core_only else list(MODEL_SPECS)


def core_model_keys() -> list[str]:
    """[Function description] core_model_keys
    - Responsibility: returns the list of default Core-4 model keys, used as the standard model set for the paper's fair benchmark.
    - Main parameters: no explicit business parameters (may only contain self/cls).
    - Returns: return type annotated as `list[str]`; see the implementation below for field meanings.
    - Note: this function mainly does in-memory computation; no extra persistence side effects beyond the called object's own behavior.
    """
    return [x.key for x in MODEL_SPECS if x.core_benchmark]


def extended_model_keys() -> list[str]:
    """[Function description] extended_model_keys
    - Responsibility: returns the extended key list of the Core-4 plus all optional models.
    - Main parameters: no explicit business parameters (may only contain self/cls).
    - Returns: return type annotated as `list[str]`; see the implementation below for field meanings.
    - Note: this function mainly does in-memory computation; no extra persistence side effects beyond the called object's own behavior.
    """
    return [x.key for x in MODEL_SPECS]
