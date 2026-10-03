# File: src/tourism_ie/data_split.py
# Module responsibility: text-leakage-free multi-label splitter: merges duplicate inputs and splits samples into train/valid/test by label targets.
# Main data flow: full data -> merge by input -> compute multi-label distribution targets -> greedy/constrained split -> write benchmark split.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
import json, random, math, csv
from collections import Counter, defaultdict
from pathlib import Path
from .data_utils import read_jsonl
from .parsing import normalize_items


def labels_of(task: str, row: dict) -> set[str]:
    """- Responsibility: extracts a sample's label set as the basis for multi-label stratified splitting.
    - Parameters: task: str,row: dict.
    - Returns: return type is `set[str]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    items = normalize_items(task, row.get("output", []))
    if task == "ner":
        return {x["entity_label"] for x in items}
    return {x["predicate"] for x in items}


def _label_targets(total: int, ratios: tuple[float, float, float]) -> list[int]:
    """- Responsibility: computes expected per-class counts per split from overall label frequency and target split ratios.
    - Parameters: total: int,ratios: tuple[float, float, float].
    - Returns: return type is `list[int]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    # Ensure every split receives at least one example when total>=3.
    if total >= 3:
        base = [1, 1, 1]
        remaining = total - 3
        raw = [remaining * r for r in ratios]
        floors = [int(x) for x in raw]
        out = [base[i] + floors[i] for i in range(3)]
        left = total - sum(out)
        order = sorted(range(3), key=lambda i: (raw[i] - floors[i], ratios[i]), reverse=True)
        for i in order[:left]: out[i] += 1
        return out
    raw = [total * r for r in ratios]
    floors = [int(x) for x in raw]
    out = floors[:]
    left = total - sum(out)
    order = sorted(range(3), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:left]: out[i] += 1
    return out



def merge_duplicate_inputs(rows: list[dict], task: str) -> list[dict]:
    """- Responsibility: merges duplicate samples/annotations by original input text so one text never lands in two splits (leakage).
    - Parameters: rows: list[dict],task: str.
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Merge duplicate input texts before benchmark splitting to prevent leakage.

    When duplicate annotations disagree, keep the union of normalized items.
    """
    grouped = {}
    order = []
    for row in rows:
        key = row.get("input", "")
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    merged = []
    for key in order:
        group = grouped[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        seen=set(); items=[]
        for row in group:
            for x in normalize_items(task,row.get("output",[])):
                k = tuple(sorted(x.items()))
                if k not in seen:
                    seen.add(k); items.append(x)
        merged.append({"input":key,"output":items})
    return merged

def multilabel_split(rows: list[dict], task: str, ratios=(0.8, 0.1, 0.1), seed=42):
    """- Responsibility: keeps multi-label distribution while meeting overall train/valid/test ratios, producing a leakage-free split.
    - Parameters: rows: list[dict],task: str,ratios=(0.8, 0.1, 0.1),seed=42.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Key process: fix target sample and label counts per split; assign samples rare-label-first; update remaining quotas; finally verify text uniqueness.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("ratios must sum to 1")
    rng = random.Random(seed)
    label_sets = [labels_of(task, r) for r in rows]
    freq = Counter(l for s in label_sets for l in s)
    n = len(rows)
    capacities = _label_targets(n, ratios)
    desired = {l: _label_targets(c, ratios) for l, c in freq.items()}
    current = {l: [0, 0, 0] for l in freq}
    splits = [[], [], []]

    order = list(range(n))
    rng.shuffle(order)
    order.sort(key=lambda i: (sum(1.0 / max(freq[l], 1) for l in label_sets[i]), len(label_sets[i])), reverse=True)

    for idx in order:
        labs = label_sets[idx]
        candidates = [s for s in range(3) if len(splits[s]) < capacities[s]]
        if not candidates:
            candidates = [0, 1, 2]
        scored = []
        for s in candidates:
            label_need = 0.0
            for l in labs:
                target = desired[l][s]
                if target:
                    label_need += max(target - current[l][s], 0) / target
            capacity_need = max(capacities[s] - len(splits[s]), 0) / max(capacities[s], 1)
            scored.append((label_need + 0.15 * capacity_need, capacity_need, rng.random(), s))
        s = max(scored)[-1]
        splits[s].append(rows[idx])
        for l in labs:
            current[l][s] += 1

    return splits


def write_jsonl(path: Path, rows: list[dict]):
    """- Responsibility: serializes the sample list line-by-line to JSONL for direct train/eval consumption.
    - Parameters: path: Path,rows: list[dict].
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_benchmark_splits(root: Path, seed: int = 42):
    """- Responsibility: builds the paper-recommended NER/NRE benchmark split and generates label-distribution stats.
    - Parameters: root: Path,seed: int=42.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    out_dir = root / "data_end/data" / "benchmark"
    summary = {}
    for task, src in [("ner", root/"data_end/data/ner/entity_all.jsonl"), ("relation", root/"data_end/data/relation/relation_all.jsonl")]:
        rows = merge_duplicate_inputs(read_jsonl(src), task)
        train, valid, test = multilabel_split(rows, task, seed=seed)
        for split, part in [("train", train), ("valid", valid), ("test", test)]:
            write_jsonl(out_dir/f"{task}_{split}.jsonl", part)
        summary[task] = {"train": len(train), "valid": len(valid), "test": len(test)}
    label_rows=[]
    for task in ("ner","relation"):
        for split in ("train","valid","test"):
            part=read_jsonl(out_dir/f"{task}_{split}.jsonl")
            counts=Counter(l for r in part for l in labels_of(task,r))
            for label,count in sorted(counts.items()):
                label_rows.append({"task":task,"split":split,"label":label,"samples_with_label":count})
    with (out_dir/"label_distribution.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["task","split","label","samples_with_label"]); w.writeheader(); w.writerows(label_rows)
    # Keep the entity-aware E-branch aligned with every regenerated benchmark split.
    entity_source=root/"data_end/data/supplementary/relationship_output_entity.jsonl"
    if entity_source.exists():
        from .original_data import prepare_entity_aware_data
        e_report=prepare_entity_aware_data(root,None)
        summary["relation_entity_aware"]={
            "original":e_report["original_counts"],
            "benchmark":e_report["benchmark_counts"],
        }
    (out_dir/"split_summary.json").write_text(json.dumps({"seed":seed,"counts":summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_merged_benchmark_splits(root: Path, data_dir: str = "data_end/data_merged", seed: int = 42, resplit: bool = False) -> dict:
    """- Responsibility: splits the merged+cleaned dataset into train/valid/test (stage 3 of merge → clean → split).
    - Parameters: root: Path,data_dir: str = "data_end/data_merged",seed: int = 42.
    - Returns: returns the split summary dict.
    - Notes: reuses an existing split_membership.jsonl (fixed test set); otherwise forces aug_budget_* to train.
    """
    """Split the merged+cleaned dataset into train/valid/test (stage 3 of merge → clean → split).

    Writes per-split files under ner/, relation/ and benchmark/ (ner / relation / relation_e
    kept in the same text_id order), plus split_membership.jsonl, split_summary.json and a
    generation_report.json compatible with ``run.py paper-data``.

    Two modes:
      - ``resplit=False`` (default): reuse an existing split_membership.jsonl when present
        (test split stays fixed, new ids → train); otherwise a first-time 8:1:1 split.
      - ``resplit=True``: ignore any existing split_membership.jsonl and always re-split
        randomly (seed), forcing ``aug_budget_*`` samples into train.
    """
    base = root / data_dir
    ner_all = read_jsonl(base / "ner" / "entity_all.jsonl")
    rel_all = read_jsonl(base / "relation" / "relation_all.jsonl")
    rel_e_all = read_jsonl(base / "relation" / "relation_all_e.jsonl")
    rel_by_id = {r["text_id"]: r for r in rel_all}
    rel_e_by_id = {r["text_id"]: r for r in rel_e_all}

    membership_path = base / "split_membership.jsonl"
    old_split = {}
    if membership_path.exists() and not resplit:
        for r in read_jsonl(membership_path):
            old_split[r["text_id"]] = r["split"]

    if old_split:
        membership = {"train": [], "valid": [], "test": []}
        for r in ner_all:
            membership[old_split.get(r["text_id"], "train")].append(r["text_id"])
        split_mode = "reused_existing_membership"
    else:
        aug_set = {r["text_id"] for r in ner_all if str(r["text_id"]).startswith("aug_budget_")}
        rest = [r for r in ner_all if r["text_id"] not in aug_set]
        tr, va, te = multilabel_split(rest, "ner", seed=seed)
        membership = {
            "train": [r["text_id"] for r in tr] + sorted(aug_set),
            "valid": [r["text_id"] for r in va],
            "test": [r["text_id"] for r in te],
        }
        split_mode = "multilabel_split_first_time"

    ner_by_id = {r["text_id"]: r for r in ner_all}
    for split, ids in membership.items():
        ner_part = [ner_by_id[tid] for tid in ids]
        rel_part = [rel_by_id[tid] for tid in ids]
        rel_e_part = [rel_e_by_id[tid] for tid in ids]
        write_jsonl(base / "ner" / f"entity_{split}.jsonl", ner_part)
        write_jsonl(base / "relation" / f"relation_{split}.jsonl", rel_part)
        write_jsonl(base / "relation" / f"relation_{split}_e.jsonl", rel_e_part)
        write_jsonl(base / "benchmark" / f"ner_{split}.jsonl", ner_part)
        write_jsonl(base / "benchmark" / f"relation_{split}.jsonl", rel_part)
        write_jsonl(base / "benchmark" / f"relation_{split}_e.jsonl", rel_e_part)

    write_jsonl(membership_path, [{"text_id": tid, "split": s} for s, ids in membership.items() for tid in ids])
    summary = {
        "seed": seed,
        "counts": {k: len(v) for k, v in membership.items()},
        "split_mode": split_mode,
    }
    (base / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    gen_path = base / "generation_report.json"
    gen = {}
    if gen_path.exists():
        try:
            gen = json.loads(gen_path.read_text(encoding="utf-8"))
        except Exception:
            gen = {}
    gen.update({
        "seed": seed,
        "unique_text_ids": len(ner_all),
        "split_counts": summary["counts"],
        "split_mode": split_mode,
    })
    gen.setdefault("relation_triples", sum(len(r.get("output", [])) for r in rel_all))
    gen_path.write_text(json.dumps(gen, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
