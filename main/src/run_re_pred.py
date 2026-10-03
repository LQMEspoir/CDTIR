# -*- coding: utf-8 -*-
"""Reuse step-2 NER predictions and directly run RE-E-Pred / RE-E-Gold.

Skip cdtir-predict's internal second NER prediction — step-2 NER and cdtir-predict's
internal NER use identical args (same model/router/weighted/top-k 0/test), so results are equivalent.

Prereqs: run_re_paper.sh steps 1-2 done (outputs/ner_paper + outputs/ner_paper_predictions.jsonl),
     step 4 RE training done (outputs/re_paper/final_adapter).

Usage:python src/run_re_pred.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tourism_ie.paper import build_pipeline_inputs, _read, _write, _predict_args
from tourism_ie.inference import predict


def main() -> None:
    ner_pred_file = ROOT / "outputs" / "ner_paper_predictions.jsonl"
    rel_test_e = ROOT / "data_end/data_merged" / "relation" / "relation_test_e.jsonl"
    re_adapter = ROOT / "outputs" / "re_paper" / "final_adapter"
    out = ROOT / "outputs" / "cdtir_pipeline"
    out.mkdir(parents=True, exist_ok=True)

    if not ner_pred_file.exists():
        sys.exit(f"missing NER predictions {ner_pred_file}; run run_re_paper.sh round 1/2 first")
    if not re_adapter.exists():
        sys.exit(f"missing RE adapter {re_adapter}; run RE training first")

    # build RE inputs from step-2 NER predicted entities (pred uses predicted entities / gold uses gold entities)
    ner = _read(ner_pred_file)
    pred_rows, gold_rows = build_pipeline_inputs(ner, _read(rel_test_e))
    pred_input = out / "re_e_pred_input.jsonl"
    gold_input = out / "re_e_gold_input.jsonl"
    _write(pred_input, pred_rows)
    _write(gold_input, gold_rows)

    common = dict(task="relation", model="qwen2.5-7b", adapter=str(re_adapter),
                  max_new_tokens=256, prompt_profile="paper_re_e")
    print("=== RE-E-Pred(NER predicted entities -> RE)===", flush=True)
    predict(ROOT, _predict_args(**common, dataset_file=str(pred_input),
                                output=str(out / "re_e_pred_predictions.jsonl")))
    print("=== RE-E-Gold(gold entities -> RE)===", flush=True)
    predict(ROOT, _predict_args(**common, dataset_file=str(gold_input),
                                output=str(out / "re_e_gold_predictions.jsonl")))

    print("\noutputs:")
    print("  re_e_pred:", out / "re_e_pred_predictions.jsonl")
    print("  re_e_gold:", out / "re_e_gold_predictions.jsonl")


if __name__ == "__main__":
    main()
