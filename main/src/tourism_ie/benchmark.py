# File: src/tourism_ie/benchmark.py
# Module responsibility: unified multi-model, multi-strategy benchmark orchestrator: composes model×strategy runs, calls train/eval and aggregates CSV/JSON.
# Main data flow: model list + strategy list + task/split -> run each experiment -> flatten metrics and resource stats -> summary/per-class aggregation.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
import csv, json, traceback
from argparse import Namespace
from pathlib import Path
from .models import resolve_model, core_model_keys, extended_model_keys
from .strategies import STRATEGIES, get_strategy
from .training import train
from .inference import predict
from .metrics import evaluate_file
from .data_split import build_benchmark_splits


def _csv_list(value: str, all_values: list[str]):
    """- Responsibility: normalizes a comma-separated CLI string into a de-spaced list.
    - Parameters: value: str,all_values: list[str].
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    v=value.strip().lower()
    if v=="all": return all_values
    if v=="extended": return extended_model_keys()
    return [x.strip() for x in value.split(",") if x.strip()]


def _flatten_result(task,model_key,strategy,train_report,pred_info,metrics,status="ok",error=""):
    """- Responsibility: flattens one run's nested metrics, config and resource stats into a single row for summary.csv.
    - Parameters: task,model_key,strategy,train_report,pred_info,metrics,status='ok',error=''.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    spec=resolve_model(model_key)
    legacy=(metrics or {}).get("legacy_text_metrics",{})
    d={
        "task":task,"model_key":spec.key,"model_name":spec.model_name,"family":spec.family,"params":spec.params,
        "strategy":strategy,"status":status,"error":error,
        "micro_precision":metrics.get("micro",{}).get("precision",0.0) if metrics else "",
        "micro_recall":metrics.get("micro",{}).get("recall",0.0) if metrics else "",
        "micro_f1":metrics.get("micro",{}).get("f1",0.0) if metrics else "",
        "macro_precision":metrics.get("macro",{}).get("precision",0.0) if metrics else "",
        "macro_recall":metrics.get("macro",{}).get("recall",0.0) if metrics else "",
        "macro_f1":metrics.get("macro",{}).get("f1",0.0) if metrics else "",
        "weighted_f1":metrics.get("weighted",{}).get("f1",0.0) if metrics else "",
        "accuracy":metrics.get("accuracy",0.0) if metrics else "",
        "exact_sample_accuracy":metrics.get("exact_sample_accuracy",0.0) if metrics else "",
        "token_accuracy":train_report.get("best_eval_token_acc","") if train_report else "",
        "bleu4":legacy.get("bleu4","") if metrics else "",
        "rougeL":legacy.get("rougeL","") if metrics else "",
        "training_seconds":train_report.get("training_seconds",0.0) if train_report else 0.0,
        "peak_gpu_memory_gb":train_report.get("peak_gpu_memory_gb",0.0) if train_report else 0.0,
        "total_parameters":train_report.get("total_parameters","") if train_report else "",
        "trainable_parameters":train_report.get("trainable_parameters","") if train_report else "",
        "trainable_percent":train_report.get("trainable_percent","") if train_report else "",
        "inference_seconds":pred_info.get("inference_seconds",0.0) if pred_info else "",
        "seconds_per_sample":pred_info.get("seconds_per_sample",0.0) if pred_info else "",
        "samples":metrics.get("samples","") if metrics else "",
    }
    return d


def _write_summary(out_dir: Path, rows: list[dict], per_class_rows: list[dict] | None = None):
    """- Responsibility: writes overall and per-class results of multiple benchmark runs into stable CSV summary files.
    - Parameters: out_dir: Path,rows: list[dict],per_class_rows: list[dict] | None=None.
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"summary.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    if rows:
        with (out_dir/"summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if per_class_rows:
        with (out_dir/"per_class_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(per_class_rows[0].keys())); w.writeheader(); w.writerows(per_class_rows)


def run_benchmark(root: Path,args):
    """- Responsibility: iterates model×strategy combos, runs training or zero-shot eval, and collects unified benchmark results.
    - Parameters: root: Path,args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Key process: resolve model and strategy sets; create a separate output dir per combo; run the zero-shot or train route; finally collect unified metrics plus time/memory info.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;involves training/backprop; memory/randomness affect resource usage and reproducibility;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    if not (root/"data_end/data/benchmark/ner_train.jsonl").exists():
        build_benchmark_splits(root,seed=args.seed)
    model_keys=_csv_list(args.models,core_model_keys())
    strategies=_csv_list(args.strategies,[k for k in STRATEGIES if k != "relation_legacy_fixed"])
    if "relation_legacy_fixed" in strategies:
        raise ValueError("relation_legacy_fixed uses a different raw SPO schema and standalone evaluator; run run_original_relation_py_fixed instead of the tourism benchmark.")
    tasks=["ner","relation"] if args.task=="both" else [args.task]
    for m in model_keys: resolve_model(m)
    for s in strategies: get_strategy(s)
    matrix=[(t,m,s) for t in tasks for m in model_keys for s in strategies]
    print("Benchmark matrix:")
    for i,(t,m,s) in enumerate(matrix,1): print(f"  {i:03d}. task={t} model={m} strategy={s}")
    if args.dry_run:
        print(f"Dry run only: {len(matrix)} experiments planned; no model was loaded.")
        return []

    base_out=Path(args.output_dir) if args.output_dir else root/"outputs/benchmark"
    rows=[]; per_class_rows=[]
    for idx,(task,model_key,strategy) in enumerate(matrix,1):
        run_id=f"{task}__{model_key.replace('/','_')}__{strategy}__seed{args.seed}"
        run_dir=base_out/"runs"/run_id; run_dir.mkdir(parents=True,exist_ok=True)
        print("\n"+"="*90); print(f"[{idx}/{len(matrix)}] {run_id}"); print("="*90)
        try:
            spec=resolve_model(model_key); strat=get_strategy(strategy)
            train_report=None; adapter=None; router_dir=None; infer_model=spec.model_name
            if strat.trainable:
                ta=Namespace(**vars(args)); ta.task=task; ta.model=model_key; ta.strategy=strategy; ta.split="benchmark"; ta.output_dir=str(run_dir/"train")
                train_report=train(root,ta)
                if strat.router: router_dir=train_report["artifact_path"]
                elif strat.peft: adapter=train_report["artifact_path"]
                else: infer_model=train_report["artifact_path"]
            pred_path=run_dir/"predictions.jsonl"
            pa=Namespace(
                task=task,model=infer_model,adapter=adapter,router_dir=router_dir,router_threshold=getattr(args,"router_threshold",None),
                router_generation_mode=getattr(args,"router_generation_mode","top1"),
                router_generation_top_k=getattr(args,"router_generation_top_k",0),
                text=None,input_file=None,output=str(pred_path),max_new_tokens=args.max_new_tokens,limit=args.eval_limit,
                split="benchmark",eval_split="test",load_in_4bit=(strategy=="qlora"),
                prompt_profile=getattr(args,"prompt_profile","canonical"),
                # Keep prediction/evaluation on the same dataset tree used by
                # training.  Without this, --data-dir data_generate trained on
                # the paper split but silently predicted against data/benchmark.
                data_dir=getattr(args,"data_dir",None),
                swanlab=False,swanlab_project=None,swanlab_experiment=None,
            )
            pred_info=predict(root,pa)
            metrics=evaluate_file(root,task,pred_path,run_dir,
                                  legacy_text_metrics=getattr(args,"legacy_text_metrics",False),
                                  plot_metrics=getattr(args,"plot_metrics",False))
            rows.append(_flatten_result(task,model_key,strategy,train_report,pred_info,metrics))
            for pc in metrics.get("per_class",[]):
                per_class_rows.append({"task":task,"model_key":spec.key,"model_name":spec.model_name,"strategy":strategy,"seed":args.seed,**pc})
        except Exception as e:
            err=f"{type(e).__name__}: {e}"
            (run_dir/"ERROR.txt").write_text(err+"\n\n"+traceback.format_exc(),encoding="utf-8")
            print(f"[FAILED] {err}")
            rows.append(_flatten_result(task,model_key,strategy,None,None,None,status="failed",error=err))
            if args.fail_fast:
                _write_summary(base_out,rows,per_class_rows); raise
        _write_summary(base_out,rows,per_class_rows)
    print(f"\nBenchmark summary: {base_out/'summary.csv'}")
    return rows
