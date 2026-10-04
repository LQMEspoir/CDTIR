# File: original_eval.py
# Module responsibility: standalone eval script matching the original project's metric: normalizes predicted/gold triples and computes the original flat metrics.
# Main data flow: JSONL predictions/gold -> triple normalization -> flatten sets -> count TP/FP/FN -> output original-metric results.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

"""
Reproduce the original qw_e.ipynb evaluation method
============================
The original code flattens each triple [ob1, rel, ob2] into an element set and compares:
    targets = list(set(sum(tt, [])))
    preds = list(set(sum(tp, [])))

Usage:
    python original_eval.py --predictions outputs/relation_predictions.jsonl --gold data/relation/relation_valid.jsonl
"""
import argparse
import json
from pathlib import Path


def load_jsonl(path):
    """[Function] load_jsonl
    - Responsibility: reads a JSONL file line-by-line into a list of dicts, skipping/handling blank lines.
    - Parameters: path.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_triplets(items):
    """[Function] normalize_triplets
    - Responsibility: normalizes different relation-triple representations into a comparable canonical structure, avoiding field-order/type differences.
    - Parameters: items.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Unify the triple format into a list of [ob1, rel, ob2]."""
    out = []
    for x in items:
        if not isinstance(x, dict):
            continue
        ob1 = x.get("subject", x.get("ob1", ""))
        rel = x.get("predicate", x.get("rel", ""))
        ob2 = x.get("object", x.get("ob2", ""))
        if ob1 and rel and ob2:
            out.append([str(ob1).strip(), str(rel).strip(), str(ob2).strip()])
    return out


def original_flatten_eval(records):
    """[Function] original_flatten_eval
    - Responsibility: counts correct/error/missed triples by the original "flattened set matching" criterion.
    - Parameters: records.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """
    Original qw_e.ipynb evaluation logic:
    flatten all samples' triples into an element set, then compute set-level P/R/F1.
    """
    all_targets = []  # flattened elements of all gold triples
    all_preds = []    # flattened elements of all predicted triples

    for r in records:
        if "gold" not in r:
            continue
        gold_items = normalize_triplets(r.get("gold", []))
        pred_items = normalize_triplets(r.get("prediction", []))

        # flatten: [[ob1,rel,ob2], [ob1,rel,ob2], ...] -> [ob1, rel, ob2, ob1, rel, ob2, ...]
        flat_gold = sum(gold_items, [])
        flat_pred = sum(pred_items, [])

        all_targets.extend(flat_gold)
        all_preds.extend(flat_pred)

    # compare after dedup (consistent with the original code)
    targets_set = set(all_targets)
    preds_set = set(all_preds)

    tp = len(targets_set & preds_set)
    fp = len(preds_set - targets_set)
    fn = len(targets_set - preds_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "method": "original_flatten (qw_e.ipynb)",
        "unique_target_elements": len(targets_set),
        "unique_pred_elements": len(preds_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    """[Function] main
    - Responsibility: script entry: parse args, run required validation, and invoke the corresponding business flow per command.
    - Parameters: no explicit business params (possibly self/cls).
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;emits logs or warnings.
    """
    parser = argparse.ArgumentParser(description="flattened evaluate method from the original qw_e.ipynb")
    parser.add_argument("--predictions", required=True, help="prediction results JSONL file")
    parser.add_argument("--gold", required=True, help="gold standard JSONL file")
    parser.add_argument("--output", default=None, help="output results JSON file path")
    args = parser.parse_args()

    preds = load_jsonl(args.predictions)
    gold = load_jsonl(args.gold)

    # merge gold into predictions
    gold_map = {}
    for g in gold:
        key = g.get("input", "")
        gold_map[key] = g.get("output", [])

    records = []
    for p in preds:
        key = p.get("input", "")
        records.append({
            "gold": gold_map.get(key, []),
            "prediction": p.get("prediction", []),
        })

    result = original_flatten_eval(records)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresults saved to: {args.output}")


if __name__ == "__main__":
    main()
