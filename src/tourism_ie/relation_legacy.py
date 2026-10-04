# File: src/tourism_ie/relation_legacy.py
# Module responsibility: fixed compatibility training route for relation.py: corrects SPO offset preprocessing, causal-LM prompt/target label alignment, and provides train/eval functions.
# Main data flow: raw SPO data -> offset-to-text triples -> fixed prompt/label encoding -> adapter training -> generate -> triple/text metrics.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import torch

LEGACY_MODEL_KEY = "qwen2-1.5b-relation-legacy"
# Legacy fault-diagnosis relation-extraction prompt. English translation:
#   "You are an expert in fault diagnosis and text entity-relation extraction.
#    Entity types: {bearing type, fault mode, fault feature, fault cause, fault location, countermeasure, preventive measure, severity, accident consequence}.
#    Predicate types: {causes, suggests, located at, prevents, characterizes, occurs at, belongs to, triggers}.
#    Extract all triples from the text below; output one triple per line as <subject, predicate, object>. Output only the extracted triples."
LEGACY_PROMPT_PREFIX = (
    "你是一个故障诊断和文本实体关系抽取领域的专家。\n"
    "现在有以下实体类型：{轴承类型, 故障模式, 故障特征, 故障原因, 故障位置, 解决对策, 预防措施, 严重程度, 事故后果}，\n"
    "predicate 类型：{导致, 建议, 位于, 预防, 表征, 发生于, 属于, 引发}。\n"
    "请从下面的文本中提取所有三元组，输出格式要求如下：每一行为一个三元组，形式为 <主语, 谓语, 宾语>。\n"
    "请不要输出其他解释性语言，仅输出提取出的三元组：\n\n"
)
TRIPLE_RE = re.compile(r"<(.+?),\s*(.+?),\s*(.+?)>")


def _source_hash(path: Path) -> str:
    """- Responsibility: computes a content hash of the source data to record the legacy training input version and improve auditability.
    - Parameters: path: Path.
    - Returns: return type is `str`; see implementation below for field details.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spo_source(path: str | Path) -> list[dict]:
    """- Responsibility: reads the SPO source used by the original relation.py and normalizes it into a preprocessable sample sequence.
    - Parameters: path: str | Path.
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Legacy relation source not found: {path}. Expected the relation.py schema: "
            "a JSON array of {'text': ..., 'spo_list': [...]} records."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Legacy relation source must be a JSON array")
    for i, row in enumerate(data):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not isinstance(row.get("spo_list"), list):
            raise ValueError(f"Invalid legacy relation row {i}: requires text:str and spo_list:list")
    return data


def _span_text(text: str, span, field: str, row_index: int) -> str:
    """- Responsibility: slices subject/object text from the sentence by character offset, handling out-of-range or abnormal spans.
    - Parameters: text: str,span,field: str,row_index: int.
    - Returns: return type is `str`; see implementation below for field details.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError(f"row {row_index}: {field} must be [start,end]")
    start, end = int(span[0]), int(span[1])
    if start < 0 or end < start or end > len(text):
        raise ValueError(f"row {row_index}: invalid {field} span {span} for text length {len(text)}")
    return text[start:end]


def preprocess_spo_entry(entry: dict, row_index: int = 0) -> dict:
    """- Responsibility: converts a single offset SPO annotation into a text triple and fixed training target.
    - Parameters: entry: dict,row_index: int=0.
    - Returns: return type is `dict`; see implementation below for field details.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    """Reproduce relation.py's offset->triple conversion and fault-diagnosis prompt."""
    text = entry["text"]
    triples = []
    for spo in entry.get("spo_list", []):
        if not isinstance(spo, dict) or "subject" not in spo or "object" not in spo or "predicate" not in spo:
            raise ValueError(f"row {row_index}: malformed SPO item: {spo!r}")
        subject = _span_text(text, spo["subject"], "subject", row_index)
        object_ = _span_text(text, spo["object"], "object", row_index)
        predicate = str(spo["predicate"])
        triples.append(f"<{subject}, {predicate}, {object_}>")
    return {
        "text": text,
        "input_text": LEGACY_PROMPT_PREFIX + text,
        "target_text": "\n".join(triples),
        "triples": [list(x) for x in extract_triples("\n".join(triples))],
    }


def preprocess_spo_rows(rows: list[dict]) -> list[dict]:
    """- Responsibility: batch-preprocesses SPO entries and filters invalid samples.
    - Parameters: rows: list[dict].
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    return [preprocess_spo_entry(row, i) for i, row in enumerate(rows)]


def prepare_relation_legacy_file(source: Path, output: Path):
    """- Responsibility: converts legacy SPO raw data and writes an intermediate file consumable by the current training entry.
    - Parameters: source: Path,output: Path.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;emits logs or warnings.
    """
    rows = preprocess_spo_rows(load_spo_source(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info = {"source": str(source), "output": str(output), "rows": len(rows), "sha256": _source_hash(source)}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def split_legacy_rows(rows: list[dict], train_ratio: float = 0.9):
    """- Responsibility: splits legacy relation samples by a fixed seed/ratio so the sets are stable across reruns.
    - Parameters: rows: list[dict],train_ratio: float=0.9.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    n_train = int(len(rows) * train_ratio)
    if len(rows) >= 2:
        n_train = max(1, min(n_train, len(rows) - 1))
    return rows[:n_train], rows[n_train:]


def encode_fixed_row(tokenizer, row: dict, max_length: int) -> dict:
    """- Responsibility: encodes a single relation sample into a causal-LM sequence, ensuring prompt tokens have labels set to -100.
    - Parameters: tokenizer,row: dict,max_length: int.
    - Returns: return type is `dict`; see implementation below for field details.
    - Notes: inference/tensor paths must keep device and dtype consistent.
    """
    """Pure row encoder used by the legacy-fixed relation dataset and smoke tests."""
    prompt_ids = tokenizer(row["input_text"], add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["target_text"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    # Keep the beginning of the original prompt and reserve room for supervision.
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[:max_length]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[: max_length - len(answer_ids)]
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def _encode_fixed(tokenizer, rows: list[dict], max_length: int):
    """- Responsibility: batch/map version of the fixed-alignment encoder, for Dataset.map.
    - Parameters: tokenizer,rows: list[dict],max_length: int.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Causal-LM-correct version of relation.py encoding.

    relation.py padded prompt and target independently to 512 and assigned target IDs as
    labels for the prompt positions. That creates positionally misaligned supervision.
    Here the exact legacy prompt/target texts are retained, but labels supervise only the
    appended target tokens: [prompt][target], labels=[-100...][target].
    """
    from datasets import Dataset
    ds = Dataset.from_list(rows)
    return ds.map(lambda row: encode_fixed_row(tokenizer,row,max_length), remove_columns=ds.column_names)


def _effective_source(root: Path, args) -> Path:
    """- Responsibility: resolves the actual legacy data-source path from user args and existing project files.
    - Parameters: root: Path,args.
    - Returns: return type is `Path`; see implementation below for field details.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    source = getattr(args, "relation_legacy_source", None)
    if source:
        return Path(source)
    packaged = root / "data_end/data" / "original_reference" / "text_entity_relation_extraction.json"
    if packaged.exists():
        return packaged
    raise SystemExit(
        "relation_legacy_fixed needs --relation-legacy-source PATH. The supplied legacy relation.py "
        "references text_entity_relation_extraction.json, but that raw fault-diagnosis dataset is not packaged here."
    )


def train_relation_legacy_fixed(root: Path, args):
    """- Responsibility: runs the fixed legacy relation LoRA training and saves the adapter, config and training stats.
    - Parameters: root: Path,args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;involves training/backprop; memory/randomness affect resource usage and reproducibility;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    """Independent, runnable reproduction of legacy/original_scripts/relation.py.

    Preserved: SPO offset conversion, fault-diagnosis prompt, Qwen2-1.5B-Instruct,
    fp16-on-CUDA, q_proj/v_proj LoRA, r=16/alpha=64/dropout=.05, 90/10 ordered split,
    lr=5e-5, 5 epochs, bs=4, accum=4, max_len=512, save_steps=100, linear scheduler.
    Fixed: decoder-only causal labels are aligned to appended target tokens.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    except ImportError as e:
        raise SystemExit("Missing training dependencies. Run: pip install -r requirements.txt") from e
    from .modeling import count_parameters
    from .models import resolve_model

    set_seed(args.seed)
    source = _effective_source(root, args)
    raw = load_spo_source(source)
    rows = preprocess_spo_rows(raw)
    if getattr(args, "limit", 0):
        rows = rows[: args.limit]
    train_rows, eval_rows = split_legacy_rows(rows, getattr(args, "relation_legacy_train_ratio", 0.9))
    if not train_rows:
        raise ValueError("No training rows after legacy relation split")

    spec = resolve_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True, use_cache=False
    )
    model.config.use_cache = False
    if use_cuda:
        model.to(torch.device("cuda"))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    targets = [x for x in ("q_proj", "v_proj") if any(n.rsplit(".", 1)[-1] == x for n, _ in model.named_modules())]
    if targets != ["q_proj", "v_proj"]:
        raise RuntimeError(f"Legacy relation.py requires q_proj/v_proj; found {targets}")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=targets, inference_mode=False,
        r=int(args.lora_r), lora_alpha=int(args.lora_alpha), lora_dropout=float(args.lora_dropout), bias="none"
    )
    model = get_peft_model(model, lora)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_ds = _encode_fixed(tokenizer, train_rows, int(args.max_length))
    eval_ds = _encode_fixed(tokenizer, eval_rows, int(args.max_length)) if eval_rows else None
    out = Path(args.output_dir) if args.output_dir else root / "outputs" / "relation_legacy_fixed_qwen2_1.5b"
    out.mkdir(parents=True, exist_ok=True)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_strategy="epoch" if eval_ds is not None else "no",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        fp16=use_cuda,
        bf16=False,
        dataloader_pin_memory=use_cuda,
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=2,
    )
    from .tracking import EvalLossCurveCallback
    loss_cb = EvalLossCurveCallback()
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[loss_cb],
    )
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start
    final = out / "final_adapter"
    trainer.model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if use_cuda else 0.0
    params = count_parameters(trainer.model)
    manifest = {
        "format": "tourism_ie_relation_legacy_fixed_v1",
        "algorithm_reference": "legacy/original_scripts/relation.py",
        "source": str(source),
        "source_sha256": _source_hash(source),
        "model_key": spec.key,
        "model_name": spec.model_name,
        "dtype": "float16" if use_cuda else "float32_cpu_fallback",
        "target_modules": targets,
        "lora_r": int(args.lora_r), "lora_alpha": int(args.lora_alpha), "lora_dropout": float(args.lora_dropout),
        "epochs": args.epochs, "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum, "learning_rate": args.learning_rate, "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length, "save_strategy": args.save_strategy, "save_steps": args.save_steps,
        "train_ratio": getattr(args, "relation_legacy_train_ratio", 0.9),
        "train_rows": len(train_rows), "eval_rows": len(eval_rows),
        "label_alignment": "fixed_prompt_plus_target_with_prompt_masked_-100",
        "legacy_bug_not_copied": "relation.py assigned independently padded target token IDs to prompt positions",
        "artifact_path": str(final), "training_seconds": elapsed, "peak_gpu_memory_gb": peak,
        "eval_losses": loss_cb.eval_losses,
        **params, "trainer_metrics": result.metrics,
    }
    (out / "legacy_relation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nLegacy relation.py fixed training completed. Saved to: {final}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "trainer_metrics"}, ensure_ascii=False, indent=2))
    return manifest


def extract_triples(text: str) -> set[tuple[str, str, str]]:
    """- Responsibility: parses generated output into a normalized triple set for structured exact-match evaluation.
    - Parameters: text: str.
    - Returns: return type is `set[tuple[str, str, str]]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    return {(a.strip(), b.strip(), c.strip()) for a, b, c in TRIPLE_RE.findall(str(text or ""))}


def triple_metrics(predictions: list[str], references: list[str]):
    """- Responsibility: computes micro precision/recall/F1 over gold/pred triple sets.
    - Parameters: predictions: list[str],references: list[str].
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        p, r = extract_triples(pred), extract_triples(ref)
        tp += len(p & r); fp += len(p - r); fn += len(r - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _text_metrics(predictions: list[str], references: list[str]):
    """- Responsibility: computes legacy text-level supplementary metrics such as BLEU/ROUGE.
    - Parameters: predictions: list[str],references: list[str].
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    try:
        import sacrebleu
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise SystemExit("Legacy relation evaluation needs requirements.txt") from e
    bleu = float(sacrebleu.corpus_bleu(predictions, [references], smooth_method="exp").score) if predictions else 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge = (sum(scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)) / len(predictions) * 100.0) if predictions else 0.0
    return {"bleu4": bleu, "rougeL": float(rouge)}


def evaluate_relation_legacy_adapter(root: Path, args):
    """- Responsibility: loads the legacy adapter, generates relation results on the eval set, and computes both triple and text metrics.
    - Parameters: root: Path,args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;involves training/backprop; memory/randomness affect resource usage and reproducibility;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit("Missing inference dependencies. Run: pip install -r requirements.txt") from e

    source = Path(args.source)
    rows = preprocess_spo_rows(load_spo_source(source))
    _, eval_rows = split_legacy_rows(rows, args.train_ratio)
    adapter = Path(args.adapter)
    peft_cfg = PeftConfig.from_pretrained(adapter)
    base_name = peft_cfg.base_model_name_or_path or "Qwen/Qwen2-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(adapter if (adapter / "tokenizer_config.json").exists() else base_name, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    use_cuda = torch.cuda.is_available(); dtype = torch.float16 if use_cuda else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, adapter)
    device = torch.device("cuda" if use_cuda else "cpu"); model.to(device); model.eval()

    predictions, references, result_rows = [], [], []
    for i in range(0, len(eval_rows), max(1, args.batch_size)):
        batch = eval_rows[i:i + max(1, args.batch_size)]
        prompts = [r["input_text"] for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        elapsed = time.perf_counter() - start
        for j, row in enumerate(batch):
            prompt_len = int(enc["attention_mask"][j].sum().item())
            # With left padding, generated sequence includes padded input width; slice by tensor width.
            gen = ids[j, enc["input_ids"].shape[1]:]
            pred = tok.decode(gen, skip_special_tokens=True).strip()
            ref = row["target_text"]
            predictions.append(pred); references.append(ref)
            result_rows.append({
                "text": row["text"], "prediction": pred, "reference": ref,
                "predicted_triples": [list(x) for x in sorted(extract_triples(pred))],
                "reference_triples": [list(x) for x in sorted(extract_triples(ref))],
                "batch_inference_seconds": elapsed,
            })

    tm = triple_metrics(predictions, references)
    textm = _text_metrics(predictions, references)
    metrics = {"triple": tm, "text": textm, "samples": len(eval_rows), "train_ratio": args.train_ratio}
    out = Path(args.output_dir) if args.output_dir else root / "outputs" / "relation_legacy_fixed_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prediction_results.json").write_text(json.dumps(result_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.plot_metrics:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise SystemExit("Plotting needs matplotlib; install requirements.txt") from e
        names = ["BLEU-4", "ROUGE-L", "Triple Precision", "Triple Recall", "Triple F1"]
        vals = [textm["bleu4"], textm["rougeL"], tm["precision"]*100, tm["recall"]*100, tm["f1"]*100]
        fig, ax = plt.subplots(figsize=(10,5)); bars = ax.bar(names, vals); ax.set_ylim(0,100); ax.set_title("Legacy relation.py fixed evaluation")
        for bar, v in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.2f}", ha="center")
        fig.tight_layout(); fig.savefig(out / "all_metrics.png", dpi=180); plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Prediction details saved to: {out}")
    return metrics
