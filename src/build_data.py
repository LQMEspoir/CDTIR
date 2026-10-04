# -*- coding: utf-8 -*-
"""Build data_end/data_merged from data_end/data_origin + the 150 budget-augmented samples.

Pipeline (data_origin → data_merged):
  1. load data_origin/entity_all.jsonl + relation_all.jsonl + relation_all_e.jsonl (flat)
  2. append the 150 aug samples from data_end/data_aug/budget_samples.jsonl to entity_all
     (and to relation_all with empty output, relation_all_e with entities + empty output)
  3. clean (apply curated NER corrections)
  4. split 8:1:1 into train/valid/test

Usage: python src/build_data.py [--resplit]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data_merged from data_origin (append → clean → split)")
    parser.add_argument("--resplit", action="store_true",
                        help="ignore the existing split_membership.jsonl and re-split randomly")
    args = parser.parse_args()

    origin = ROOT / "data_end" / "data_origin"
    merged = ROOT / "data_end" / "data_merged"
    aug_file = ROOT / "data_end" / "data_aug" / "budget_samples.jsonl"

    # 1. load origin (flat) and append the 150 augmented samples
    ner = _read(origin / "entity_all.jsonl")
    rel = _read(origin / "relation_all.jsonl")
    rel_e = _read(origin / "relation_all_e.jsonl")
    aug = _read(aug_file)
    existing = {r["text_id"] for r in ner}
    new_aug = [a for a in aug if a["text_id"] not in existing]
    ner += new_aug
    rel += [{"text_id": a["text_id"], "input": a["input"], "output": []} for a in new_aug]
    rel_e += [{"text_id": a["text_id"], "input": a["input"], "entities": a["output"], "output": []} for a in new_aug]
    print(f"[1/3] append aug: {len(new_aug)} samples → entity_all {len(ner)} rows")

    # 2. write merged (nested)
    _write(merged / "ner" / "entity_all.jsonl", ner)
    _write(merged / "relation" / "relation_all.jsonl", rel)
    _write(merged / "relation" / "relation_all_e.jsonl", rel_e)

    # 3. clean + split (reuse the modular steps)
    from tourism_ie.data_cleaning import clean_merged_data
    from tourism_ie.data_split import build_merged_benchmark_splits
    clean_report = clean_merged_data(ROOT, "data_end/data_merged")
    print(f"[2/3] clean: applied={clean_report['applied']}, rows={clean_report['counts']['all']}")
    split_report = build_merged_benchmark_splits(ROOT, "data_end/data_merged", resplit=args.resplit)
    print(f"[3/3] split: {split_report['counts']} ({split_report['split_mode']})")
    print("done → data_end/data_merged")


if __name__ == "__main__":
    main()
