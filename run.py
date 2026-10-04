# File: run.py
# Module responsibility: unified CLI entry: registers train/predict/evaluate/benchmark/smoke-test subcommands and dispatches parsed args to tourism_ie implementations.
# Main data flow: CLI args -> subcommand parsing -> validation/default fill -> call train/inference/eval/benchmark -> write results or print to terminal.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Runtime artifacts belong to the directory from which the command is run.
# Set CDTIR_ROOT only when a caller intentionally wants a different workspace.
# Keep the code location separate so ``python /path/to/run.py`` still imports
# the package while writing data, checkpoints and reports to the current dir.
CODE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CDTIR_ROOT", Path.cwd())).expanduser().resolve()
SRC = CODE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def add_tracking_args(p):
    """[Function] add_tracking_args
    - Responsibility: registers experiment-tracking / SwanLab CLI options on argparse subparsers.
    - Parameters: p.
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    p.add_argument("--swanlab", action="store_true", help="Enable optional SwanLab experiment tracking")
    p.add_argument("--swanlab-project", default=None)
    p.add_argument("--swanlab-experiment", default=None)


def add_prompt_args(p):
    """[Function] add_prompt_args
    - Responsibility: registers prompt-related options (prompt profile, relation input mode) on the CLI parser.
    - Parameters: p.
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    p.add_argument("--prompt-profile", choices=["canonical","original","original_strict","entity_aware","text_only_strict","entity_aware_strict","paper_ner","paper_re_e","paper_re_e_v2","paper_re_e_v3","paper_re_eh","paper_re_to"], default="canonical",
                   help="original closely follows prompts in the supplied legacy scripts/notebooks; entity_aware is E-branch RE (text + entities); text_only_strict / entity_aware_strict reproduce original train_t.json / train_e.json byte-for-byte in a single user message")
    p.add_argument("--target-format", choices=["canonical","original"], default="canonical",
                   help="original reproduces legacy NER concatenated-JSON / NRE ob1-rel-ob2 targets")


def add_training_args(p, include_model=True, include_strategy=True):
    """[Function] add_training_args
    - Responsibility: centrally registers training hyperparams, LoRA/QLoRA, router, focal-loss and resource args so subcommands share consistent config.
    - Parameters: p,include_model=True,include_strategy=True.
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    if include_model:
        p.add_argument("--model", default="qwen2.5-1.5b", help="Registry key, Hugging Face model name, or local model directory")
    if include_strategy:
        from tourism_ie.strategies import STRATEGIES
        p.add_argument("--strategy", choices=list(STRATEGIES), default="lora")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-length", type=int, default=512)
    # None lets task-specific original-compatible defaults be selected automatically.
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--focal-alpha", type=float, default=0.25, help="Original entity_QW2.5_fl.py default")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Original entity_QW2.5_fl.py default")
    p.add_argument("--balance-max-multiplier", type=int, default=5)
    p.add_argument("--empty-relation-weight", type=float, default=1.0,
                   help="Per-sample loss weight for empty-relation samples (relation task only); 1.0=off, e.g. 0.1 down-weights 'no relation' samples")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-strategy", choices=["no","steps","epoch"], default="epoch",
                   help="Checkpoint cadence; original NER scripts used steps with --save-steps 100")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-strategy", choices=["no","steps","epoch"], default="epoch",
                   help="Evaluation cadence; strict tourism RE notebooks used steps")
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--lr-scheduler-type", default="linear", help="Transformers LR scheduler; strict tourism RE notebooks used cosine")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-total-limit", type=int, default=2)
    # grouped_router_multilora compatibility knobs
    p.add_argument("--router-adapter-epochs", type=float, default=1.0, help="Epochs per named adapter for grouped_router_multilora")
    p.add_argument("--router-threshold", type=float, default=0.45, help="Confidence fallback threshold for grouped_router_multilora")
    # original-compatible neural Router-MultiLoRA knobs
    p.add_argument("--router-loss-weight", type=float, default=0.5, help="Original joint router BCE coefficient")
    p.add_argument("--router-forward-top-k", type=int, default=0, help="0=all label adapters (faithful); >0 reduces compute and is approximate")
    p.add_argument("--router-checkpoint-mixture", action=argparse.BooleanOptionalAction, default=True,
                   help="Checkpoint the multi-adapter mixture to reduce training memory")
    p.add_argument("--router-hidden-mode", choices=["frozen_base","strict_original"], default="frozen_base",
                   help="frozen_base (paper default) routes detached base-Qwen hidden states; strict_original is retained only for legacy comparison")
    p.add_argument("--max-steps", type=int, default=-1, help="Positive values override epochs; use 5 for a quick smoke run")
    p.add_argument("--resume-from-checkpoint", default=None, help="latest or an explicit checkpoint path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0=all training samples")
    p.add_argument("--data-dir", default=None,
                   help="Optional data tree containing ner/, relation/, and benchmark/ (for example data_generate)")
    add_prompt_args(p)
    add_tracking_args(p)


def main():
    """[Function] main
    - Responsibility: script entry: parse args, run required validation, and invoke the corresponding business flow per command.
    - Parameters: no explicit business params (possibly self/cls).
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;involves training/backprop; memory/randomness affect resource usage and reproducibility;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    parser = argparse.ArgumentParser(description="Tourism IE benchmark: NER + relation extraction")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke-test", help="Validate package/data/metrics without downloading a model")
    p.add_argument("--verbose", action="store_true")

    sub.add_parser("list-models", help="Show Core-4 benchmark backbones plus optional/legacy registered models")
    sub.add_parser("list-strategies", help="Show supported training strategies")

    p = sub.add_parser("split-data", help="Split the merged+cleaned dataset into 80/10/10 benchmark splits (merge → clean → split)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data_end/data_merged")
    p.add_argument("--resplit", action="store_true", help="Ignore any existing split_membership.jsonl and re-split randomly")

    p = sub.add_parser("clean-data", help="Apply curated NER corrections to the merged dataset (merge → clean → split)")
    p.add_argument("--data-dir", default="data_end/data_merged")

    p = sub.add_parser("prepare-original-data", help="Rebuild canonical NER/NRE JSONL from packaged original source files")
    p.add_argument("--ner-source", default=None)
    p.add_argument("--relation-source", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("prepare-entity-aware-re", help="Build runnable Entity-aware RE JSONL; uses train_e/valid_e when present, otherwise packaged relationship_output_entity.jsonl")
    p.add_argument("--source-dir", default=None, help="Optional directory containing original train_e.json and valid_e.json; packaged supplementary data is the automatic fallback")
    p.add_argument("--output-dir", default=None, help="Output directory for original-split E-branch files (default: data/relation); benchmark E-branch files are also rebuilt")

    p = sub.add_parser("train", help="Fine-tune one model with a selected strategy")
    p.add_argument("--task", choices=["ner", "relation"], required=True)
    p.add_argument("--split", choices=["original","benchmark"], default="original")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--relation-legacy-source", default=None,
                   help="Raw JSON array with text+spo_list for relation_legacy_fixed (legacy relation.py schema)")
    p.add_argument("--relation-legacy-train-ratio", type=float, default=0.9)
    add_training_args(p)

    p = sub.add_parser("prepare-relation-legacy", help="Convert legacy relation.py text+SPO offsets into <subject, predicate, object> prompt/target JSONL")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("relation-legacy-eval", help="Evaluate a relation_legacy_fixed adapter on the held-out tail split")
    p.add_argument("--adapter", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--plot-metrics", action="store_true")

    p = sub.add_parser("predict", help="Run NER or relation inference")
    p.add_argument("--task", choices=["ner", "relation"], required=True)
    p.add_argument("--model", default="qwen2.5-1.5b")
    p.add_argument("--adapter", default=None, help="Optional LoRA/QLoRA adapter directory")
    p.add_argument("--router-dir", default=None, help="Router-MultiLoRA package directory; overrides --adapter when set")
    p.add_argument("--router-threshold", type=float, default=None, help="Grouped-router threshold override")
    p.add_argument("--router-generation-mode", choices=["top1","weighted"], default="top1",
                   help="Neural router: top1 is practical; weighted reproduces original weighted-logit idea during greedy generation")
    p.add_argument("--router-generation-top-k", type=int, default=0, help="weighted mode: 0=all adapters; >0 approximate top-k mixture")
    p.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit for QLoRA adapter inference")
    p.add_argument("--split", choices=["original","benchmark"], default="original")
    p.add_argument("--eval-split", choices=["valid","test"], default="test")
    p.add_argument("--text", default=None)
    p.add_argument("--entities-json", default=None, help="For a single Entity-aware RE --text: JSON list of entity_text/entity_label objects")
    p.add_argument("--input-file", default=None, help="UTF-8 .txt, one sample per line")
    p.add_argument("--dataset-file", default=None, help="JSONL rows with text_id/input/output and optional entities")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prompt-profile", choices=["canonical","original","original_strict","entity_aware","text_only_strict","entity_aware_strict","paper_ner","paper_re_e","paper_re_e_v2","paper_re_e_v3","paper_re_eh","paper_re_to"], default="canonical")
    add_tracking_args(p)

    p = sub.add_parser("vllm-predict", help="Use an already running vLLM OpenAI-compatible server, matching the original notebook workflow")
    p.add_argument("--task", choices=["ner","relation"], required=True)
    p.add_argument("--served-model", required=True, help="Base model name or --lora-modules alias served by vLLM")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--split", choices=["original","benchmark"], default="original")
    p.add_argument("--eval-split", choices=["valid","test"], default="test")
    p.add_argument("--text", default=None)
    p.add_argument("--entities-json", default=None, help="For a single Entity-aware RE --text: JSON list of entity_text/entity_label objects")
    p.add_argument("--input-file", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prompt-profile", choices=["canonical","original","original_strict","entity_aware","text_only_strict","entity_aware_strict","paper_ner","paper_re_e","paper_re_e_v2","paper_re_e_v3","paper_re_eh","paper_re_to"], default="original")

    p = sub.add_parser("evaluate", help="Overall + per-class P/R/F1/Presence-Acc evaluation")
    p.add_argument("--task", choices=["ner", "relation"], required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--legacy-text-metrics", action="store_true", help="Also compute BLEU-4 + ROUGE-L like the original relation.py")
    p.add_argument("--plot-metrics", action="store_true", help="Save all_metrics.png (requires matplotlib)")

    p = sub.add_parser("benchmark", help="Run Core-4 backbone and/or multi-strategy comparison experiments")
    p.add_argument("--task", choices=["ner","relation","both"], default="both")
    p.add_argument("--models", default="all", help="all=Core-4; extended=all registered models; or comma-separated registry keys")
    p.add_argument("--strategies", default="lora", help="Comma-separated strategy keys; use list-strategies to inspect")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--eval-limit", type=int, default=0, help="0=evaluate full benchmark test split")
    p.add_argument("--dry-run", action="store_true", help="Print experiment matrix without loading models")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--legacy-text-metrics", action="store_true")
    p.add_argument("--plot-metrics", action="store_true")
    p.add_argument("--router-generation-mode", choices=["top1","weighted"], default="top1")
    p.add_argument("--router-generation-top-k", type=int, default=0)
    add_training_args(p, include_model=False, include_strategy=False)

    p = sub.add_parser("paper-data", help="Validate the immutable-ID paper dataset and regenerate audit/manifest files")
    p.add_argument("--data-dir", default="data_end/data_merged")

    p = sub.add_parser("paper-train", help="Train a fixed CDTIR paper preset")
    p.add_argument("--variant", choices=["CDTIR","CDTIR_NMoRE","CDTIR_F","CDTIR_EH","CDTIR_TO"], required=True)
    p.add_argument("--task", choices=["ner","relation","both"], default="both")
    p.add_argument("--model", default=None, help="Override only for smoke tests; paper results require qwen2.5-7b")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--empty-relation-weight", type=float, default=1.0,
                   help="Per-sample loss weight for empty-relation samples (relation task only); 1.0=off")

    p = sub.add_parser("cdtir-predict", help="Run NER -> RE-E-Pred and the separate RE-E-Gold upper bound")
    p.add_argument("--variant", choices=["CDTIR","CDTIR_NMoRE","CDTIR_F"], default="CDTIR")
    p.add_argument("--model", default="qwen2.5-7b")
    p.add_argument("--ner-router", default=None)
    p.add_argument("--ner-adapter", default=None)
    p.add_argument("--re-adapter", default=None, help="Required for trained variants; omit for frozen CDTIR_F")
    p.add_argument("--ner-prompt-profile", default="paper_ner", help="NER prompt profile (zero-shot: use canonical's concise prompt)")
    p.add_argument("--re-prompt-profile", default="paper_re_e", help="RE prompt profile (e.g. paper_re_e_v2)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)

    p = sub.add_parser("paper-report", help="Build Figure 4, Table 9 and Table 10 from real histories/predictions")
    p.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    def cli_has(flag: str) -> bool:
        """[Function] main.cli_has
        - Responsibility: runs the local "cli has" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: flag: str.
        - Returns: return type is `bool`; see implementation below for field details.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        return any(x == flag or x.startswith(flag + "=") for x in sys.argv[1:])

    # A bare relation_legacy_fixed invocation should reproduce relation.py hyperparameters.
    # Explicit CLI values still win.
    if args.command == "train" and getattr(args, "strategy", None) == "relation_legacy_fixed":
        if getattr(args, "task", None) != "relation":
            parser.error("relation_legacy_fixed requires --task relation")
        legacy_defaults = {
            "--model": ("model", "qwen2-1.5b-relation-legacy"),
            "--epochs": ("epochs", 5.0),
            "--batch-size": ("batch_size", 4),
            "--eval-batch-size": ("eval_batch_size", 4),
            "--grad-accum": ("grad_accum", 4),
            "--learning-rate": ("learning_rate", 5e-5),
            "--warmup-ratio": ("warmup_ratio", 0.0),
            "--max-length": ("max_length", 512),
            "--lora-r": ("lora_r", 16),
            "--lora-alpha": ("lora_alpha", 64),
            "--lora-dropout": ("lora_dropout", 0.05),
            "--logging-steps": ("logging_steps", 1),
            "--save-strategy": ("save_strategy", "steps"),
            "--save-steps": ("save_steps", 100),
        }
        for flag, (attr, value) in legacy_defaults.items():
            if not cli_has(flag):
                setattr(args, attr, value)
    if args.command == "smoke-test":
        from tourism_ie.smoke import run_smoke_test
        raise SystemExit(0 if run_smoke_test(ROOT, verbose=args.verbose) else 1)
    if args.command == "paper-data":
        from tourism_ie.paper import validate_paper_data
        print(json.dumps(validate_paper_data(ROOT,args.data_dir),ensure_ascii=False,indent=2)); return
    if args.command == "paper-train":
        from tourism_ie.paper import paper_train
        print(json.dumps(paper_train(ROOT,args),ensure_ascii=False,indent=2,default=float)); return
    if args.command == "cdtir-predict":
        from tourism_ie.paper import cdtir_predict
        if args.variant != "CDTIR_F" and not args.re_adapter:
            parser.error("cdtir-predict requires --re-adapter except for CDTIR_F")
        print(json.dumps(cdtir_predict(ROOT,args),ensure_ascii=False,indent=2)); return
    if args.command == "paper-report":
        from tourism_ie.paper import paper_report
        print(json.dumps(paper_report(ROOT,args),ensure_ascii=False,indent=2)); return
    if args.command == "list-models":
        from tourism_ie.models import list_models
        print(json.dumps([x.to_dict() for x in list_models()],ensure_ascii=False,indent=2)); return
    if args.command == "list-strategies":
        from tourism_ie.strategies import STRATEGIES
        print(json.dumps({k:v.__dict__ for k,v in STRATEGIES.items()},ensure_ascii=False,indent=2)); return
    if args.command == "split-data":
        from tourism_ie.data_split import build_merged_benchmark_splits
        print(json.dumps(build_merged_benchmark_splits(ROOT,args.data_dir,args.seed,resplit=args.resplit),ensure_ascii=False,indent=2)); return
    if args.command == "clean-data":
        from tourism_ie.data_cleaning import clean_merged_data
        print(json.dumps(clean_merged_data(ROOT,args.data_dir),ensure_ascii=False,indent=2)); return
    if args.command == "prepare-original-data":
        from tourism_ie.original_data import prepare_original_data
        print(json.dumps(prepare_original_data(ROOT,args),ensure_ascii=False,indent=2)); return
    if args.command == "prepare-entity-aware-re":
        from tourism_ie.original_data import prepare_entity_aware_data
        print(json.dumps(prepare_entity_aware_data(ROOT,args),ensure_ascii=False,indent=2)); return
    if args.command == "train":
        from tourism_ie.training import train
        train(ROOT, args); return
    if args.command == "prepare-relation-legacy":
        from tourism_ie.relation_legacy import prepare_relation_legacy_file
        prepare_relation_legacy_file(Path(args.source), Path(args.output)); return
    if args.command == "relation-legacy-eval":
        from tourism_ie.relation_legacy import evaluate_relation_legacy_adapter
        evaluate_relation_legacy_adapter(ROOT, args); return
    if args.command == "predict":
        from tourism_ie.inference import predict
        predict(ROOT, args); return
    if args.command == "vllm-predict":
        from tourism_ie.vllm_inference import predict_vllm
        predict_vllm(ROOT,args); return
    if args.command == "evaluate":
        from tourism_ie.metrics import evaluate_file
        evaluate_file(ROOT,args.task,Path(args.predictions),Path(args.output_dir) if args.output_dir else None,
                      legacy_text_metrics=args.legacy_text_metrics,plot_metrics=args.plot_metrics); return
    if args.command == "benchmark":
        from tourism_ie.benchmark import run_benchmark
        run_benchmark(ROOT,args); return

if __name__ == "__main__":
    main()
