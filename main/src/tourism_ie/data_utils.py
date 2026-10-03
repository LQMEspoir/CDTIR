# File: src/tourism_ie/data_utils.py
# Module responsibility: data access and prompt construction utilities shared by training and inference. Unified JSONL reading, task paths, NER/NRE input formats, label extraction, and balanced sampling.
# Main data flow: task/data profile + raw rows -> standard messages/prompt -> training target text/labels; also provides path resolution and oversampling helpers.
# Reading suggestion: start with the public functions/classes in this file, then follow imports to shared modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments and does not modify any existing expressions, control flow, parameter defaults, or function call relationships.

from __future__ import annotations
import json, math
from collections import Counter
from pathlib import Path
from .parsing import normalize_items
from .prompts import system_prompt, RELATION_TEXT_ONLY_STRICT_USER, RELATION_ENTITY_AWARE_STRICT_USER


def read_jsonl(path: Path) -> list[dict]:
    """- Responsibility: read a JSONL data file and return a list of the JSON objects from each line.
    - Main parameters: path: Path.
    - Returns: return type annotated as `list[dict]`; see the implementation below for field meanings.
    - Note: may read/write disk files/model weights, so ensure the output directory is writable; includes explicit parameter/data validation and raises on invalid input.
    """
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except Exception as e: raise ValueError(f"Invalid JSONL: {path}:{line_no}: {e}") from e
    return rows


def is_entity_aware_profile(prompt_profile: str) -> bool:
    """- Responsibility: determine whether the current relation extraction prompt/profile requires "text + known entities" input.
    - Main parameters: prompt_profile: str.
    - Returns: return type annotated as `bool`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    return prompt_profile in {"entity_aware", "entity_aware_strict", "paper_re_e", "paper_re_e_v2", "paper_re_e_v3"}


def is_strict_relation_profile(prompt_profile: str) -> bool:
    """- Responsibility: determine whether the current relation extraction enables the strict/legacy-aligned profiles.
    - Main parameters: prompt_profile: str.
    - Returns: return type annotated as `bool`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    return prompt_profile in {"text_only_strict", "entity_aware_strict"}


def task_paths(root: Path, task: str, split_scheme: str = "original", prompt_profile: str = "canonical", data_dir: str | Path | None = None):
    """- Responsibility: resolve the standard file paths for the training set, validation set, etc. based on the task type and split profile.
    - Main parameters: root: Path, task: str, split_scheme: str='original', prompt_profile: str='canonical'.
    - Returns: returns the processed object/metric/tensor/path result; the actual structure is determined by each return branch.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    data_root = Path(data_dir) if data_dir else root/"data_end/data"
    entity_aware = task == "relation" and is_entity_aware_profile(prompt_profile)
    suffix = "_e" if entity_aware else ""
    if split_scheme == "benchmark":
        base = data_root/"benchmark"
        return base/f"{task}_train{suffix}.jsonl", base/f"{task}_valid{suffix}.jsonl"
    if task == "ner":
        return data_root/"ner/entity_train.jsonl", data_root/"ner/entity_test.jsonl"
    return data_root/f"relation/relation_train{suffix}.jsonl", data_root/f"relation/relation_valid{suffix}.jsonl"


def eval_path(root: Path, task: str, split_scheme: str = "original", eval_split: str = "test", prompt_profile: str = "canonical", data_dir: str | Path | None = None):
    """- Responsibility: resolve the data file that should currently be evaluated based on task, split, and eval_split.
    - Main parameters: root: Path, task: str, split_scheme: str='original', eval_split: str='test', prompt_profile: str='canonical'.
    - Returns: returns the processed object/metric/tensor/path result; the actual structure is determined by each return branch.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    entity_aware = task == "relation" and is_entity_aware_profile(prompt_profile)
    suffix = "_e" if entity_aware else ""
    if split_scheme == "benchmark":
        split = "test" if eval_split == "test" else "valid"
        base=Path(data_dir) if data_dir else root/"data_end/data"
        return base/"benchmark"/f"{task}_{split}{suffix}.jsonl"
    return task_paths(root, task, split_scheme, prompt_profile, data_dir)[1]


def target_text(task: str, row: dict, target_format: str = "canonical", prompt_profile: str = "canonical") -> str:
    """- Responsibility: serialize structured NER/NRE annotations into the target text used for supervised learning.
    - Main parameters: task: str, row: dict, target_format: str='canonical', prompt_profile: str='canonical'.
    - Returns: return type annotated as `str`; see the implementation below for field meanings.
    - Note: may read/write disk files/model weights, so ensure the output directory is writable.
    """
    value=normalize_items(task,row.get("output",[]))
    # Paper targets are part of the experiment contract and must not vary with
    # the generic --target-format flag.  Table 4-6 requires one valid JSON
    # object per NER output line; Table 4-7 requires ob1/rel/ob2 JSON or the
    # explicit no-relation sentence.
    if task == "ner" and prompt_profile == "paper_ner":
        if not value:
            return "没有找到任何实体"
            # return "no entity found"
        return "\n".join(json.dumps(x, ensure_ascii=False) for x in value)
    if task == "relation" and prompt_profile in {"paper_re_e", "paper_re_e_v2", "paper_re_e_v3", "paper_re_eh", "paper_re_to"}:
        if not value:
            return "找不到关系"
            # return "no relation found"
        legacy=[{"ob1":x["subject"],"rel":x["predicate"],"ob2":x["object"]} for x in value]
        return json.dumps(legacy, ensure_ascii=False, indent=2)
    # The original qwen-t/qw_e Swift datasets stored assistant targets as a
    # pretty-printed ob1/rel/ob2 JSON list wrapped in a ```json code fence.
    # Strict RE prompt profiles imply that exact target representation so a
    # caller cannot accidentally pair a strict legacy input with canonical labels.
    if task == "relation" and is_strict_relation_profile(prompt_profile):
        legacy=[{"ob1":x["subject"],"rel":x["predicate"],"ob2":x["object"]} for x in value]
        return "```json\n" + json.dumps(legacy,ensure_ascii=False,indent=2) + "\n```"
    if target_format == "original":
        if task == "ner":
            if not value:
                return "没有找到任何实体"
            # Original NER scripts trained on consecutive JSON objects rather than an array.
            return "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in value)
        legacy=[{"ob1":x["subject"],"rel":x["predicate"],"ob2":x["object"]} for x in value]
        return json.dumps(legacy, ensure_ascii=False, separators=(",", ":")) if legacy else "找不到关系"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _format_entities(entities: list[dict], compact: bool = False) -> str:
    """- Responsibility: format the entity list into a stable, readable text representation for entity-aware relation extraction prompts.
    - Main parameters: entities: list[dict], compact: bool=False.
    - Returns: return type annotated as `str`; see the implementation below for field meanings.
    - Note: may read/write disk files/model weights, so ensure the output directory is writable.
    """
    """Format the original concatenated entity objects used by the E-branch."""
    if not entities:
        return ""
    parts = []
    for e in entities:
        text = e.get("entity_text", "")
        label = e.get("entity_label", "")
        if text and label:
            kwargs={"ensure_ascii":False}
            if compact:
                kwargs["separators"]=(",", ":")
            parts.append(json.dumps({"entity_text": text, "entity_label": label}, **kwargs))
    return "".join(parts)


def _text_only_source(text: str) -> str:
    """- Responsibility: construct a relation extraction user input that contains only the original text.
    - Main parameters: text: str.
    - Returns: return type annotated as `str`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    """Return exactly one legacy ``文本:`` prefix for strict notebook prompts."""
    raw=(text or "").strip()
    if "\n实体:" in raw:
        # if "\nentities:" in raw:
        raw=raw.split("\n实体:",1)[0].rstrip()
    return raw if raw.startswith("文本:") else f"文本:{raw}"
    # return raw if raw.startswith("text:") else f"text:{raw}"


def _entity_aware_source(text: str, entities: list[dict] | None) -> str:
    """- Responsibility: construct a "text + entity list" relation extraction user input so the model predicts relations given known entities.
    - Main parameters: text: str, entities: list[dict] | None.
    - Returns: return type annotated as `str`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    base=_text_only_source(text)
    entity_str=_format_entities(entities or [],compact=True)
    return f"{base}\n实体:{entity_str}"
    # return f"{base}\nentities:{entity_str}"


def messages_for(task: str, text: str, prompt_profile: str = "canonical", entities: list[dict] | None = None) -> list[dict]:
    """- Responsibility: assemble system/user/assistant messages based on the task and profile, serving as the unified input for the chat template.
    - Main parameters: task: str, text: str, prompt_profile: str='canonical', entities: list[dict] | None=None.
    - Returns: return type annotated as `list[dict]`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    # Strict RE notebook profiles are genuinely single-user-message prompts.
    # Do not add an empty system turn: Qwen chat templates tokenize that turn.
    if task == "relation" and prompt_profile == "text_only_strict":
        content=RELATION_TEXT_ONLY_STRICT_USER.replace("{text}",_text_only_source(text))
        return [{"role":"user","content":content}]
    if task == "relation" and prompt_profile == "entity_aware_strict":
        content=RELATION_ENTITY_AWARE_STRICT_USER.replace("{text}",_entity_aware_source(text,entities))
        return [{"role":"user","content":content}]
    user_content = text
    if task == "relation" and prompt_profile in ("paper_re_e", "paper_re_e_v2", "paper_re_e_v3"):
        normalized=[{"entity_text":e.get("entity_text",""),"entity_label":e.get("entity_label","")}
                    for e in (entities or []) if e.get("entity_text") and e.get("entity_label")]
        raw=(text or "").strip()
        if raw.startswith("文本:"):
            # if raw.startswith("text:"):
            raw=raw[len("文本:"):].strip()
        user_content=(f"请处理以下文本：{raw}\n"
                      # f"please process the following text: {raw}\n"
                      f"文本对应实体如下：{json.dumps(normalized,ensure_ascii=False)}")
                      # f"entities for the text are as follows: {json.dumps(normalized,ensure_ascii=False)}")
    elif task == "relation" and prompt_profile == "entity_aware":
        entity_str = _format_entities(entities)
        user_content = f"{_text_only_source(text)}\n实体:[{entity_str}]" if entity_str else f"{_text_only_source(text)}\n实体:[]"
        # user_content = f"{_text_only_source(text)}\nentities:[{entity_str}]" if entity_str else f"{_text_only_source(text)}\nentities:[]"
    return [{"role":"system","content":system_prompt(task,prompt_profile)},{"role":"user","content":user_content}]


def _chat_template_supports_system(tokenizer) -> bool:
    """- Responsibility: probe whether the tokenizer's chat template allows starting with a system role (e.g., some models raise an exception directly).
    - Main parameters: tokenizer.
    - Returns: return type annotated as `bool`; see the implementation below for field meanings.
    - Note: the result is cached on the tokenizer instance to avoid re-rendering the probe template for every sample.
    """
    cached = getattr(tokenizer, "_cdtir_supports_system", None)
    if cached is not None:
        return cached
    template = getattr(tokenizer, "chat_template", None)
    supports = True
    if template:
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "probe"}, {"role": "user", "content": "probe"}],
                tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            supports = False
    try:
        setattr(tokenizer, "_cdtir_supports_system", supports)
    except Exception:
        pass
    return supports


def _fold_unsupported_system_role(tokenizer, messages: list[dict]) -> list[dict]:
    """- Responsibility: when the chat template does not support the system role, fold the system content into the immediately following user message to avoid rendering errors.
    - Main parameters: tokenizer, messages: list[dict].
    - Returns: return type annotated as `list[dict]`; see the implementation below for field meanings.
    - Note: only rewrites when the first message is indeed system and the template does not support it; otherwise returns unchanged.
    """
    if not messages or messages[0].get("role") != "system":
        return messages
    if _chat_template_supports_system(tokenizer):
        return messages
    system = messages[0].get("content", "")
    rest = messages[1:]
    if not system:
        return rest
    if rest and rest[0].get("role") == "user":
        rest[0] = {"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}
        return rest
    return [{"role": "user", "content": system}] + rest


def render_prompt(tokenizer, task: str, text: str, prompt_profile: str = "canonical", add_generation_prompt: bool = True, entities: list[dict] | None = None) -> str:
    """- Responsibility: apply the tokenizer chat template to messages to produce the final prompt string usable directly for training/inference.
    - Main parameters: tokenizer, task: str, text: str, prompt_profile: str='canonical', add_generation_prompt: bool=True, entities: list[dict] | None=None.
    - Returns: return type annotated as `str`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    """Render a model prompt, including strict legacy framing when requested."""
    if task == "ner" and prompt_profile == "original_strict":
        from .prompts import NER_ORIGINAL_STRICT_SYSTEM_PROMPT
        suffix = "<|im_start|>assistant\n" if add_generation_prompt else ""
        return (
            f"<|im_start|>system\n{NER_ORIGINAL_STRICT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n" + suffix
        )
    messages = _fold_unsupported_system_role(tokenizer, messages_for(task, text, prompt_profile, entities))
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=False,  # disable Qwen3 thinking mode; other models' chat templates ignore this parameter
    )



def require_entity_aware_rows(rows: list[dict], prompt_profile: str, context: str = "data") -> None:
    """- Responsibility: validate that entity-aware relation data actually contains the required entity fields, avoiding silently degrading to text-only semantics.
    - Main parameters: rows: list[dict], prompt_profile: str, context: str='data'.
    - Returns: return type annotated as `None`; see the implementation below for field meanings.
    - Note: includes explicit parameter/data validation and raises on invalid input.
    """
    """Fail loudly instead of silently degrading Entity-aware RE to text-only RE."""
    if not is_entity_aware_profile(prompt_profile):
        return
    # An empty list is a valid negative example: some texts genuinely contain no
    # entity and therefore cannot contain a relation.  Only a missing/null or
    # malformed field would silently degrade the entity-aware prompt.
    missing=[
        i for i,r in enumerate(rows)
        if "entities" not in r or r.get("entities") is None or not isinstance(r.get("entities"), list)
    ]
    if missing:
        raise ValueError(
            f"{context}: Entity-aware profile requires an entities list; "
            f"missing or malformed in {len(missing)}/{len(rows)} rows (first index={missing[0]})"
        )

def row_labels(task: str, row: dict) -> set[str]:
    """- Responsibility: extract the set of classes in a single record, used for balanced sampling and Router target construction.
    - Main parameters: task: str, row: dict.
    - Returns: return type annotated as `set[str]`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    items = normalize_items(task, row.get("output", []))
    key = "entity_label" if task == "ner" else "predicate"
    return {x[key] for x in items if x.get(key)}


def balance_rows(task: str, rows: list[dict], max_multiplier: int = 5) -> list[dict]:
    """- Responsibility: apply controlled oversampling to training samples based on class rarity, alleviating under-learning of long-tail classes during LoRA fine-tuning.
    - Main parameters: task: str, rows: list[dict], max_multiplier: int=5.
    - Returns: return type annotated as `list[dict]`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the objects it calls, it has no additional persistent side effects.
    """
    """Deterministic rare-class oversampling for multi-label extraction rows."""
    label_freq = Counter(l for r in rows for l in row_labels(task, r))
    if not label_freq:
        return list(rows)
    max_freq = max(label_freq.values())
    balanced=[]
    for row in rows:
        labs=row_labels(task,row)
        if not labs:
            mult=1
        else:
            rarest=min(label_freq[l] for l in labs)
            mult=max(1, min(max_multiplier, int(math.ceil(math.sqrt(max_freq/max(rarest,1))))))
        balanced.extend([row]*mult)
    return balanced
