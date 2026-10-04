# File: src/tourism_ie/data_cleaning.py
# Module responsibility: NER data cleaning and audit: dedup, apply manual correction rules, validate sample format and save cleaned artifacts.
# Main data flow: raw NER rows + corrections -> normalize/dedup -> apply corrections -> validate -> output cleaned data and audit records.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from .data_split import build_benchmark_splits
from .data_utils import read_jsonl
from .parsing import normalize_items
from .prompts import NER_LABELS


def _key(item: dict) -> tuple[str, str]:
    """- Responsibility: builds a stable hashable key for entity/relation items, for exact-match, dedup and set operations.
    - Parameters: item: dict.
    - Returns: return type is `tuple[str, str]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    return (str(item.get("entity_text", "")), str(item.get("entity_label", "")))


def _dedupe(items: Iterable[dict]) -> list[dict]:
    """- Responsibility: removes duplicate annotations by canonical key while preserving first-occurrence order.
    - Parameters: items: Iterable[dict].
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    out, seen = [], set()
    for item in items:
        k = _key(item)
        if k in seen:
            continue
        seen.add(k)
        out.append({"entity_text": k[0], "entity_label": k[1]})
    return out


def load_corrections(root: Path) -> list[dict]:
    """- Responsibility: loads the human-curated correction file and builds fast-lookup correction rules.
    - Parameters: root: Path.
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    path = root / "data_end/data" / "cleaning" / "ner_corrections.json"
    return json.loads(path.read_text(encoding="utf-8"))


def apply_ner_corrections(rows: list[dict], corrections: list[dict]) -> tuple[list[dict], list[dict]]:
    """- Responsibility: applies manual correction rules to NER samples and returns corrected data plus auditable change records.
    - Parameters: rows: list[dict],corrections: list[dict].
    - Returns: return type is `tuple[list[dict], list[dict]]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Apply curated, exact-input corrections and return an audit trail.

    The operation is deterministic and idempotent: each record is normalized to a
    list of entity dictionaries, exact duplicates are removed, and curated patches
    are applied only when their input key matches.
    """
    by_input = {c["match_input"]: c for c in corrections}
    cleaned, audit = [], []
    for row in rows:
        rec = deepcopy(row)
        rec["output"] = _dedupe(normalize_items("ner", rec.get("output", [])))
        corr = by_input.get(rec.get("input", ""))
        if corr:
            before_input = rec["input"]
            before = list(rec["output"])
            remove = {_key(x) for x in corr.get("remove", [])}
            rec["output"] = [x for x in rec["output"] if _key(x) not in remove]
            rec["output"].extend(corr.get("add", []))
            rec["output"] = _dedupe(rec["output"])
            if corr.get("new_input"):
                rec["input"] = corr["new_input"]
            audit.append({
                "correction_id": corr["id"],
                "before_input": before_input,
                "after_input": rec["input"],
                "before_output": before,
                "after_output": rec["output"],
                "reason": corr.get("reason", ""),
                "confidence": corr.get("confidence", ""),
            })
        cleaned.append(rec)
    return cleaned, audit


def validate_ner_rows(rows: list[dict]) -> dict:
    """- Responsibility: checks NER rows' required fields, entity structure and value validity, surfacing dirty data early.
    - Parameters: rows: list[dict].
    - Returns: return type is `dict`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    invalid_labels = []
    punctuation_only = []
    merged_values = []
    duplicates = []
    for idx, row in enumerate(rows, 1):
        seen = set()
        for item in normalize_items("ner", row.get("output", [])):
            k = _key(item)
            if item.get("entity_label") not in NER_LABELS:
                invalid_labels.append((idx, item))
            text = str(item.get("entity_text", "")).strip()
            if text and not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text):
                punctuation_only.append((idx, item))
            if ";" in text or "；" in text:
                merged_values.append((idx, item))
            if k in seen:
                duplicates.append((idx, item))
            seen.add(k)
    return {
        "rows": len(rows),
        "invalid_labels": invalid_labels,
        "punctuation_only": punctuation_only,
        "merged_values": merged_values,
        "duplicate_annotations": duplicates,
    }


def _write_jsonl(path: Path, rows: list[dict]):
    """- Responsibility: writes a dict sequence to disk as UTF-8 JSONL, ensuring the parent dir exists.
    - Parameters: path: Path,rows: list[dict].
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_ner_data(root: Path, rebuild_benchmark: bool = True, seed: int = 42) -> dict:
    """- Responsibility: chains dedup, manual correction, validity checks and audit output into cleaned benchmark-ready NER data.
    - Parameters: root: Path,rebuild_benchmark: bool=True,seed: int=42.
    - Returns: return type is `dict`; see implementation below for field details.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;includes explicit parameter/data validation; raises on invalid input.
    """
    cdir = root / "data_end/data" / "cleaning"
    sources = {
        "all": cdir / "entity_all_preclean.jsonl",
        "train": cdir / "entity_train_preclean.jsonl",
        "test": cdir / "entity_test_preclean.jsonl",
    }
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing immutable pre-clean source(s): " + ", ".join(missing))
    corrections = load_corrections(root)
    outputs = {
        "all": root / "data_end/data" / "ner" / "entity_all.jsonl",
        "train": root / "data_end/data" / "ner" / "entity_train.jsonl",
        "test": root / "data_end/data" / "ner" / "entity_test.jsonl",
    }
    audit_all = []
    validation = {}
    counts = {}
    for split in ("all", "train", "test"):
        rows = read_jsonl(sources[split])
        cleaned, audit = apply_ner_corrections(rows, corrections)
        _write_jsonl(outputs[split], cleaned)
        audit_all.extend({"split": split, **x} for x in audit)
        validation[split] = validate_ner_rows(cleaned)
        counts[split] = len(cleaned)

    audit_json = cdir / "applied_corrections.json"
    audit_json.write_text(json.dumps(audit_all, ensure_ascii=False, indent=2), encoding="utf-8")
    with (cdir / "applied_corrections.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["split", "correction_id", "before_input", "after_input", "before_output", "after_output", "reason", "confidence"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in audit_all:
            flat = dict(row)
            flat["before_output"] = json.dumps(flat["before_output"], ensure_ascii=False)
            flat["after_output"] = json.dumps(flat["after_output"], ensure_ascii=False)
            w.writerow(flat)

    serializable_validation = {
        split: {
            k: (v if k == "rows" else [{"row": i, "item": item} for i, item in v])
            for k, v in report.items()
        }
        for split, report in validation.items()
    }
    result = {"counts": counts, "applied": len(audit_all), "validation": serializable_validation}
    if rebuild_benchmark:
        result["benchmark"] = build_benchmark_splits(root, seed=seed)
    (cdir / "cleaning_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def clean_merged_data(root: Path, data_dir: str = "data_end/data_merged") -> dict:
    """- Responsibility: applies the curated NER corrections to the merged dataset (stage 2 of merge → clean → split).
    - Parameters: root: Path,data_dir: str = "data_end/data_merged".
    - Returns: return type is `dict`; see implementation below for field details.
    - Notes: may read/write disk files; idempotent (corrections are exact-input matches).
    """
    """Apply curated NER corrections to the merged dataset (stage 2 of merge → clean → split).

    Reads ner/entity_all.jsonl, relation/relation_all.jsonl and relation/relation_all_e.jsonl
    from ``data_dir``, applies the same immutable corrections used for the canonical tree,
    then propagates any input/entity changes to the relation files and writes everything back.
    """
    base = root / data_dir
    corrections = load_corrections(root)

    ner = read_jsonl(base / "ner" / "entity_all.jsonl")
    rel = read_jsonl(base / "relation" / "relation_all.jsonl")
    rel_e = read_jsonl(base / "relation" / "relation_all_e.jsonl")
    rel_by_id = {r["text_id"]: r for r in rel}
    rel_e_by_id = {r["text_id"]: r for r in rel_e}

    cleaned, audit = apply_ner_corrections(ner, corrections)
    for row in cleaned:
        tid = row["text_id"]
        if tid in rel_by_id:
            rel_by_id[tid]["input"] = row["input"]
        if tid in rel_e_by_id:
            rel_e_by_id[tid]["input"] = row["input"]
            rel_e_by_id[tid]["entities"] = row["output"]

    _write_jsonl(base / "ner" / "entity_all.jsonl", cleaned)
    _write_jsonl(base / "relation" / "relation_all.jsonl", [rel_by_id[r["text_id"]] for r in cleaned])
    _write_jsonl(base / "relation" / "relation_all_e.jsonl", [rel_e_by_id[r["text_id"]] for r in cleaned])

    cdir = base / "cleaning"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "applied_corrections.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    with (cdir / "applied_corrections.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["correction_id", "before_input", "after_input", "before_output", "after_output", "reason", "confidence"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in audit:
            flat = dict(row)
            flat["before_output"] = json.dumps(flat["before_output"], ensure_ascii=False)
            flat["after_output"] = json.dumps(flat["after_output"], ensure_ascii=False)
            w.writerow(flat)

    validation = validate_ner_rows(cleaned)
    result = {
        "counts": {"all": len(cleaned)},
        "applied": len(audit),
        "validation": {k: (v if k == "rows" else len(v)) for k, v in validation.items()},
    }
    (cdir / "cleaning_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
