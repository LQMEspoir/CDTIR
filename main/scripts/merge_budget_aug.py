#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge the budget-augmented samples from data_aug/budget_samples.jsonl into data_expanded.

Budget augmentation only generates budget entities (NER), not relations, so:
- append into ner/entity_all.jsonl (with entity output)
- append into relation/relation_all.jsonl (empty output)

Idempotent: skip if text_id already exists. After merging, rerun src/merge_trees.py to rebuild data_merged,
augmented samples are assigned to the train split because they are not in data_expanded/split_membership.jsonl.

Usage:python scripts/merge_budget_aug.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_end" / "data_expanded"
SRC = ROOT / "data_end" / "data_aug" / "budget_samples.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    aug = read(SRC)
    if not aug:
        print(f"no augmented samples: {SRC}")
        return

    ner = read(DATA / "ner" / "entity_all.jsonl")
    rel = read(DATA / "relation" / "relation_all.jsonl")
    existing = {r["text_id"] for r in ner}

    new_ner = [r for r in aug if r["text_id"] not in existing]
    new_rel = [{"text_id": r["text_id"], "input": r["input"], "output": []} for r in new_ner]

    print(f"augmented samples {len(aug)}  rows, of which new {len(new_ner)}  rows")
    if not new_ner:
        print("all already exist, nothing to merge.")
        return

    write(DATA / "ner" / "entity_all.jsonl", ner + new_ner)
    write(DATA / "relation" / "relation_all.jsonl", rel + new_rel)
    print(f"merged into data_expanded: entity_all {len(ner)}->{len(ner) + len(new_ner)}, "
          f"relation_all {len(rel)}->{len(rel) + len(new_rel)}")
    print("next: python src/merge_trees.py rebuild data_merged (augmented samples go to train)")


if __name__ == "__main__":
    main()
