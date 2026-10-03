# File: src/tourism_ie/evaluation/metrics.py
# Module responsibility: the core of fine-grained structured evaluation. Computes exact-match entity/relation Micro, Macro, Weighted P/R/F1, exact sample accuracy, and per-class Presence Accuracy.
# Main data flow: gold/pred records + label set -> class-level TP/FP/FN/TN -> aggregated metrics -> serializable metrics dict.
# Reading guide: start with this file's public functions/classes, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments; it does not modify existing expressions, control flow, default parameter values, or function call relationships.

from __future__ import annotations
from collections import defaultdict
from .labels import labels_for_task
from ..parsing import normalize_items


def _class(task: str, x: dict) -> str:
    """[Function description] _class
    - Responsibility: normalizes the category name of a single item, compatible with the different field names that entity and relation entries may use.
    - Main parameters: task: str, x: dict.
    - Returns: the return type is annotated as `str`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the called objects, it has no extra persistence side effects.
    """
    return x.get("entity_label", "") if task == "ner" else x.get("predicate", "")


def _key(task: str, x: dict):
    """[Function description] _key
    - Responsibility: constructs a stable, hashable key for entity/relation entries, used for exact-match, deduplication, and set operations.
    - Main parameters: task: str, x: dict.
    - Returns: returns processed objects/metrics/tensors/paths, etc.; the actual structure is decided by each return branch.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the called objects, it has no extra persistence side effects.
    """
    if task == "ner":
        return (x.get("entity_text", ""), x.get("entity_label", ""))
    return (x.get("subject", ""), x.get("predicate", ""), x.get("object", ""))


def _safe_div(a, b):
    """[Function description] _safe_div
    - Responsibility: performs division safely; returns 0 when the denominator is zero, avoiding ZeroDivisionError caused by unsupported categories.
    - Main parameters: a, b.
    - Returns: returns processed objects/metrics/tensors/paths, etc.; the actual structure is decided by each return branch.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the called objects, it has no extra persistence side effects.
    """
    return a / b if b else 0.0


def evaluate_records_detailed(task: str, records: list[dict]) -> dict:
    """[Function description] evaluate_records_detailed
    - Responsibility: the core fine-grained evaluator: counts exact-match TP/FP/FN and presence TN per sample/per class, and aggregates various P/R/F1/Accuracy metrics.
    - Main parameters: task: str, records: list[dict].
    - Returns: the return type is annotated as `dict`; see the implementation below for field meanings.
    - Key process: first convert gold/pred items into sets; accumulate exact-match TP/FP/FN per class; then accumulate presence TP/FP/FN/TN based on whether the class appears in the sample; finally do micro/macro/weighted aggregation.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the called objects, it has no extra persistence side effects.
    """
    labels = labels_for_task(task)
    stats = {l: {"tp":0,"fp":0,"fn":0,"support":0,"presence_tp":0,"presence_fp":0,"presence_fn":0,"presence_tn":0} for l in labels}
    micro = {"tp":0,"fp":0,"fn":0}
    exact_correct = 0
    n = 0

    for r in records:
        if "gold" not in r:
            continue
        pred_items = normalize_items(task, r.get("prediction", []))
        gold_items = normalize_items(task, r.get("gold", []))
        p_all = {_key(task, x) for x in pred_items}
        g_all = {_key(task, x) for x in gold_items}
        micro["tp"] += len(p_all & g_all)
        micro["fp"] += len(p_all - g_all)
        micro["fn"] += len(g_all - p_all)
        exact_correct += int(p_all == g_all)
        n += 1

        for label in labels:
            p = {_key(task, x) for x in pred_items if _class(task, x) == label}
            g = {_key(task, x) for x in gold_items if _class(task, x) == label}
            s = stats[label]
            s["tp"] += len(p & g)
            s["fp"] += len(p - g)
            s["fn"] += len(g - p)
            s["support"] += len(g)
            gp, pp = bool(g), bool(p)
            if gp and pp: s["presence_tp"] += 1
            elif (not gp) and pp: s["presence_fp"] += 1
            elif gp and (not pp): s["presence_fn"] += 1
            else: s["presence_tn"] += 1

    per_class = []
    for label in labels:
        s = stats[label]
        p = _safe_div(s["tp"], s["tp"] + s["fp"])
        r = _safe_div(s["tp"], s["tp"] + s["fn"])
        f1 = _safe_div(2*p*r, p+r)
        accuracy = _safe_div(s["tp"], s["tp"] + s["fp"] + s["fn"])
        presence_acc = _safe_div(s["presence_tp"] + s["presence_tn"], n)
        per_class.append({
            "label": label, "support": s["support"], "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": p, "recall": r, "f1": f1, "accuracy": accuracy, "presence_accuracy": presence_acc,
            "presence_tp": s["presence_tp"], "presence_fp": s["presence_fp"],
            "presence_fn": s["presence_fn"], "presence_tn": s["presence_tn"],
        })

    mp = _safe_div(micro["tp"], micro["tp"] + micro["fp"])
    mr = _safe_div(micro["tp"], micro["tp"] + micro["fn"])
    mf = _safe_div(2*mp*mr, mp+mr)
    # Accuracy = sum_i TP_i / (sum_i TP_i + sum_i FP_i + sum_i FN_i)
    micro_accuracy = _safe_div(micro["tp"], micro["tp"] + micro["fp"] + micro["fn"])
    active = [x for x in per_class if x["support"] > 0]
    macro_p = sum(x["precision"] for x in active)/len(active) if active else 0.0
    macro_r = sum(x["recall"] for x in active)/len(active) if active else 0.0
    macro_f = sum(x["f1"] for x in active)/len(active) if active else 0.0
    total_support = sum(x["support"] for x in active)
    weighted_p = _safe_div(sum(x["precision"]*x["support"] for x in active), total_support)
    weighted_r = _safe_div(sum(x["recall"]*x["support"] for x in active), total_support)
    weighted_f = _safe_div(sum(x["f1"]*x["support"] for x in active), total_support)
    return {
        "samples": n,
        "micro": {**micro, "precision":mp, "recall":mr, "f1":mf},
        "accuracy": micro_accuracy,
        "macro": {"precision":macro_p, "recall":macro_r, "f1":macro_f},
        "weighted": {"precision":weighted_p, "recall":weighted_r, "f1":weighted_f},
        "exact_sample_accuracy": _safe_div(exact_correct, n),
        "exact_correct_samples": exact_correct,
        "per_class": per_class,
    }
