# File: src/tourism_ie/router_grouped.py
# Module responsibility: stable Grouped-Router-MultiLoRA: routes via char TF-IDF + LogisticRegression to semantic-group adapters, falling back to general on low confidence.
# Main data flow: training rows -> labels mapped to semantic groups -> per-group LoRA + text-classifier router -> save package; at inference text -> group -> adapter.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .data_utils import read_jsonl, task_paths, row_labels, require_entity_aware_rows
from .modeling import load_tokenizer, load_base_model, find_lora_targets, count_parameters
from .models import resolve_model

ROUTER_GROUPS = {
    "ner": {
        "geo": {"目的地", "出发地", "返程地"},
        # "geo": {"Destination", "Origin", "Return"}
        "service": {"住宿", "餐饮", "产品", "交通"},
        # "service": {"Lodging", "Dining", "Product", "Transport"}
        "constraint": {"预算", "时长", "时间", "人数"},
        # "constraint": {"Budget", "Duration", "Time", "People"}
        "attribute": {"人群", "强度", "人气", "天气"},
        # "attribute": {"Crowd", "Intensity", "Popularity", "Weather"}
    },
    "relation": {
        "sequence_time": {"游玩顺序", "出行时间", "停留时长"},
        # "sequence_time": {"Play Order", "Travel Time", "Stay Duration"}
        "budget": {"预算限制", "人均预算"},
        # "budget": {"Budget Limit", "Per-capita Budget"}
        "experience": {"体验内容"},
        # "experience": {"Experience"}
    },
}


def _safe_import_router_deps():
    """- Responsibility: lazy-imports scikit-learn/joblib grouped-router deps so they aren't required unless the strategy is used.
    - Parameters: no explicit business params (possibly self/cls).
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as e:
        raise SystemExit(
            "router_multilora requires scikit-learn + joblib. Install: pip install -r requirements.txt"
        ) from e
    return joblib, TfidfVectorizer, LogisticRegression, Pipeline


def _primary_group(task: str, labels: set[str]) -> str:
    """- Responsibility: maps a sample's fine-grained labels to the router's primary semantic group, for training the group-level text classifier.
    - Parameters: task: str,labels: set[str].
    - Returns: return type is `str`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    groups = ROUTER_GROUPS[task]
    scored = []
    for order, (name, members) in enumerate(groups.items()):
        scored.append((len(labels & members), -order, name))
    best = max(scored)
    return best[2] if best[0] > 0 else "general"


def _group_rows(task: str, rows: list[dict], group_name: str) -> list[dict]:
    """- Responsibility: splits training samples by semantic group, preparing per-group LoRA adapter training data.
    - Parameters: task: str,rows: list[dict],group_name: str.
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    if group_name == "general":
        return list(rows)
    members = ROUTER_GROUPS[task][group_name]
    return [r for r in rows if row_labels(task, r) & members]


def _fit_router(task: str, rows: list[dict], out_path: Path):
    """- Responsibility: fits a lightweight text router with char TF-IDF features and balanced LogisticRegression.
    - Parameters: task: str,rows: list[dict],out_path: Path.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    joblib, TfidfVectorizer, LogisticRegression, Pipeline = _safe_import_router_deps()
    texts, targets = [], []
    for row in rows:
        group = _primary_group(task, row_labels(task, row))
        if group == "general":
            continue
        texts.append(row["input"])
        targets.append(group)
    if len(set(targets)) < 2:
        raise RuntimeError(f"Router needs at least two route classes; got {sorted(set(targets))}")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
    ])
    start = time.perf_counter()
    pipe.fit(texts, targets)
    elapsed = time.perf_counter() - start
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    return {"rows": len(texts), "classes": [str(x) for x in pipe.classes_], "training_seconds": elapsed}


def _locate_saved_adapter(root: Path, adapter_name: str) -> Path:
    """- Responsibility: locates the actually saved PEFT adapter subdir in the training output, tolerating different Trainer save layouts.
    - Parameters: root: Path,adapter_name: str.
    - Returns: return type is `Path`; see implementation below for field details.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    candidates = [root / adapter_name, root]
    for c in candidates:
        if (c / "adapter_config.json").exists():
            return c
    found = list(root.rglob("adapter_config.json"))
    if not found:
        raise RuntimeError(f"PEFT did not produce adapter_config.json for {adapter_name} under {root}")
    # When a named adapter is saved, PEFT commonly creates a named subdirectory.
    exact = [p.parent for p in found if p.parent.name == adapter_name]
    return exact[0] if exact else found[0].parent


def train_router_multilora(root: Path, args):
    """- Responsibility: trains the Router-MultiLoRA package: prepares category/group data, trains multiple adapters + router, and saves a unified manifest.
    - Parameters: root: Path,args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Key process: build category/group adapters; train router supervision; save each adapter, router weights and manifest so the package can be fully restored.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;involves training/backprop; memory/randomness affect resource usage and reproducibility;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    """Train a practical named-adapter Multi-LoRA router.

    One base model is loaded once. A general LoRA plus task-specific named LoRA
    adapters are trained sequentially. A TF-IDF logistic router predicts which
    named adapter should be activated at inference; low-confidence examples fall
    back to the general adapter. This avoids the unsafe repeated wrapping pattern
    in the legacy router script while preserving the intended routing idea.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, set_seed
    except ImportError as e:
        raise SystemExit("Missing training dependencies. Run: pip install -r requirements.txt") from e
    from .training import _encode_dataset, _effective_lora_hparams
    from .tracking import EvalLossCurveCallback

    set_seed(args.seed)
    prompt_profile = getattr(args, "prompt_profile", "canonical")
    train_path, eval_path = task_paths(root, args.task, args.split, prompt_profile, getattr(args, "data_dir", None))
    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)
    require_entity_aware_rows(train_rows,prompt_profile,"grouped-router training data")
    require_entity_aware_rows(eval_rows,prompt_profile,"grouped-router evaluation data")
    if args.limit:
        train_rows = train_rows[:args.limit]
        eval_rows = eval_rows[:max(1, min(args.limit // 5 or 1, len(eval_rows)))]

    spec = resolve_model(args.model)
    tokenizer = load_tokenizer(args.model)
    base = load_base_model(args.model, training=True, quantized=False)
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    targets = find_lora_targets(base, spec.target_modules)
    lora_r, lora_alpha, lora_dropout = _effective_lora_hparams(args)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
    )
    model = get_peft_model(base, lora, adapter_name="general")
    for group_name in ROUTER_GROUPS[args.task]:
        model.add_adapter(group_name, lora)

    out = Path(args.output_dir) if args.output_dir else root / "outputs" / f"{args.task}_{spec.key}_grouped_router_multilora"
    out.mkdir(parents=True, exist_ok=True)
    adapters_root = out / "adapters"
    adapters_root.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    adapter_reports = []
    total_training_seconds = 0.0
    adapter_names = ["general", *ROUTER_GROUPS[args.task].keys()]
    for adapter_name in adapter_names:
        rows = _group_rows(args.task, train_rows, adapter_name)
        eval_part = _group_rows(args.task, eval_rows, adapter_name)
        if not rows:
            continue
        if not eval_part:
            eval_part = eval_rows[: min(8, len(eval_rows))]
        model.set_adapter(adapter_name)
        train_ds = _encode_dataset(tokenizer, args.task, rows, args.max_length, prompt_profile, getattr(args, "target_format", "canonical"))
        eval_ds = _encode_dataset(tokenizer, args.task, eval_part, args.max_length, prompt_profile, getattr(args, "target_format", "canonical"))
        run_dir = out / "adapter_training" / adapter_name
        targs = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=getattr(args, "router_adapter_epochs", 1.0),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_strategy="no",
            eval_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=args.gradient_checkpointing,
            bf16=bf16,
            fp16=fp16,
            dataloader_pin_memory=use_cuda,
            seed=args.seed,
            data_seed=args.seed,
        )
        loss_cb = EvalLossCurveCallback()
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
            callbacks=[loss_cb],
        )
        start = time.perf_counter()
        result = trainer.train()
        elapsed = time.perf_counter() - start
        total_training_seconds += elapsed

        save_root = adapters_root / f"_save_{adapter_name}"
        model.save_pretrained(save_root, selected_adapters=[adapter_name])
        actual = _locate_saved_adapter(save_root, adapter_name)
        final_dir = adapters_root / adapter_name
        final_dir.mkdir(parents=True, exist_ok=True)
        # Copy only the lightweight adapter files to a stable path.
        import shutil
        for p in actual.iterdir():
            if p.is_file():
                shutil.copy2(p, final_dir / p.name)
        shutil.rmtree(save_root, ignore_errors=True)
        adapter_reports.append({
            "adapter": adapter_name,
            "train_rows": len(rows),
            "eval_rows": len(eval_part),
            "training_seconds": elapsed,
            "eval_losses": loss_cb.eval_losses,
            "trainer_metrics": result.metrics,
            "path": str(final_dir),
        })

    tokenizer.save_pretrained(out / "tokenizer")
    router_report = _fit_router(args.task, train_rows, out / "router.joblib")
    peak_gb = (torch.cuda.max_memory_allocated() / 1024**3) if use_cuda else 0.0
    params = count_parameters(model)
    manifest = {
        "format": "tourism_ie_grouped_router_multilora_v1",
        "task": args.task,
        "model_key": spec.key,
        "model_name": spec.model_name,
        "strategy": "grouped_router_multilora",
        "groups": {k: sorted(v) for k, v in ROUTER_GROUPS[args.task].items()},
        "adapters": {x["adapter"]: str(Path("adapters") / x["adapter"]) for x in adapter_reports},
        "general_adapter": "general",
        "router_file": "router.joblib",
        "router_threshold": float(getattr(args, "router_threshold", 0.45)),
        "prompt_profile": getattr(args, "prompt_profile", "canonical"),
        "target_format": getattr(args, "target_format", "canonical"),
        "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
        "router_report": router_report,
        "adapter_reports": adapter_reports,
        "training_seconds": total_training_seconds,
        "peak_gpu_memory_gb": peak_gb,
        **params,
    }
    (out / "router_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    report = {
        "task": args.task,
        "model_key": spec.key,
        "model_name": spec.model_name,
        "strategy": "grouped_router_multilora",
        "split": args.split,
        "seed": args.seed,
        "train_rows_original": len(train_rows),
        "train_rows_effective": sum(x["train_rows"] for x in adapter_reports),
        "eval_rows": len(eval_rows),
        "epochs": getattr(args, "router_adapter_epochs", 1.0),
        "training_seconds": total_training_seconds,
        "peak_gpu_memory_gb": peak_gb,
        "artifact_path": str(out),
        **params,
        "router_classes": router_report["classes"],
        "adapter_count": len(adapter_reports),
    }
    (out / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nGrouped Router-MultiLoRA training completed. Saved to: {out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def load_router_package(router_dir: str | Path, quantized: bool = False):
    """- Responsibility: reads router manifest, tokenizer, base/PEFT model and router weights from disk to rebuild an inference-ready package.
    - Parameters: router_dir: str | Path,quantized: bool=False.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: involves training/backprop; memory/randomness affect resource usage and reproducibility;inference/tensor paths must keep device and dtype consistent.
    """
    joblib, *_ = _safe_import_router_deps()
    from peft import PeftModel

    router_dir = Path(router_dir)
    manifest = json.loads((router_dir / "router_manifest.json").read_text(encoding="utf-8"))
    base_name = manifest["model_name"]
    tok = load_tokenizer(base_name)
    base = load_base_model(base_name, training=False, quantized=quantized)
    adapters = manifest["adapters"]
    first = manifest.get("general_adapter", "general")
    first_path = router_dir / adapters[first]
    model = PeftModel.from_pretrained(base, first_path, adapter_name=first, is_trainable=False)
    for name, rel in adapters.items():
        if name == first:
            continue
        model.load_adapter(router_dir / rel, adapter_name=name, is_trainable=False)
    router = joblib.load(router_dir / manifest["router_file"])
    if not getattr(model, "hf_device_map", None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model, router, manifest, device


def route_text(text: str, router, manifest: dict, threshold: float | None = None) -> tuple[str, float]:
    """- Responsibility: computes router probabilities from the text and picks the target category/group, applying top-k or low-confidence fallback.
    - Parameters: text: str,router,manifest: dict,threshold: float | None=None.
    - Returns: return type is `tuple[str, float]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    probs = router.predict_proba([text])[0]
    classes = [str(x) for x in router.classes_]
    best_i = max(range(len(probs)), key=lambda i: probs[i])
    confidence = float(probs[best_i])
    chosen = classes[best_i]
    threshold = manifest.get("router_threshold", 0.45) if threshold is None else threshold
    if confidence < threshold or chosen not in manifest["adapters"]:
        chosen = manifest.get("general_adapter", "general")
    return chosen, confidence
