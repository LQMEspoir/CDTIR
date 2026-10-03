# -*- coding: utf-8 -*-
"""Merge data_end/data_generate + data_end/data_expanded into data_end/data_merged (merge only, no split).

Dedup rule: for overlapping text_ids keep the data_end/data_generate version (old gold wins).

Writes only ner/entity_all.jsonl, relation/relation_all.jsonl and relation/relation_all_e.jsonl.
Cleaning and splitting are separate steps in the pipeline (merge → clean → split):
    python src/merge_trees.py    # 1. merge
    python run.py clean-data     # 2. clean
    python run.py split-data     # 3. split

Usage: python src/merge_trees.py [output_dir]
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "data_end/data_generate"
NEW = ROOT / "data_end/data_expanded"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data_end/data_merged"

    # merge NER + relations (no split - clean & split are separate steps)
    old_ner = load(OLD / "ner" / "entity_all.jsonl")
    new_ner = load(NEW / "ner" / "entity_all.jsonl")
    old_ids = {r["text_id"] for r in old_ner}
    new_dedup = [r for r in new_ner if r["text_id"] not in old_ids]
    print(f"data_end/data_generate {len(old_ner)} + data_end/data_expanded {len(new_ner)}, "
          f"dedup {len(new_ner) - len(new_dedup)} rows -> merged {len(old_ner) + len(new_dedup)} rows")

    ner_all = old_ner + new_dedup
    rel_by_id = {r["text_id"]: r["output"] for r in load(OLD / "relation" / "relation_all.jsonl")}
    for r in load(NEW / "relation" / "relation_all.jsonl"):
        if r["text_id"] not in old_ids:
            rel_by_id[r["text_id"]] = r["output"]

    by_id = {r["text_id"]: r for r in ner_all}
    relation_by_tid = {tid: {"text_id": tid, "input": by_id[tid]["input"], "output": rel_by_id[tid]} for tid in by_id}
    entity_aware_by_tid = {tid: {"text_id": tid, "input": by_id[tid]["input"], "entities": by_id[tid]["output"],
                                  "output": rel_by_id[tid]} for tid in by_id}

    write_jsonl(out_dir / "ner" / "entity_all.jsonl", ner_all)
    write_jsonl(out_dir / "relation" / "relation_all.jsonl", [relation_by_tid[r["text_id"]] for r in ner_all])
    write_jsonl(out_dir / "relation" / "relation_all_e.jsonl", [entity_aware_by_tid[r["text_id"]] for r in ner_all])

    report = {
        "unique_text_ids": len(ner_all),
        "source_counts": {
            "data_generate": len(old_ner),
            "data_expanded": len(new_ner),
            "dedup_removed": len(new_ner) - len(new_dedup),
        },
        "relation_triples": sum(len(rel_by_id[tid]) for tid in rel_by_id),
    }
    (out_dir / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nmerged (not split) → {out_dir}\nnext: python run.py clean-data && python run.py split-data")


if __name__ == "__main__":
    main()
