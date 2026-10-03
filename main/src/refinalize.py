# -*- coding: utf-8 -*-
"""Regenerate the whole tree after human review of entity_all.jsonl / relation_all.jsonl.

Read ner/entity_all.jsonl (entity truth) + relation/relation_all.jsonl (relation truth) from data_end/data_expanded,
reuse finalize to re-split 8:1:1 and rewrite relation_all_e / train/valid/test / benchmark / split_membership.

Usage: python refinalize.py [data_dir]
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data_end/data_expanded"

    ner_rows = load(base / "ner" / "entity_all.jsonl")
    rel_rows = load(base / "relation" / "relation_all.jsonl")
    rel_by_id = {r["text_id"]: r["output"] for r in rel_rows}

    missing = [r["text_id"] for r in ner_rows if r["text_id"] not in rel_by_id]
    if missing:
        sys.exit(f"relation file missing {len(missing)} text_ids, please fill them first: {missing[:3]}")

    input_by_id = {r["text_id"]: r["input"] for r in ner_rows}
    generated = {
        tid: {"text_id": tid, "input": input_by_id[tid], "output": rel_by_id[tid],
              "source": "human_reviewed", "rejected": []}
        for tid in input_by_id
    }

    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_relation_truth_ollama_with_ner as g
    report = g.finalize(base, ner_rows, generated, seed=42, model="qwen2.5:7b")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nre-split and written back → {base}")


if __name__ == "__main__":
    main()
