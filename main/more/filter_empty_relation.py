# -*- coding: utf-8 -*-
"""
Filter empty-relation negatives from the relation training data in data_end/data_generate, keeping only samples with relations (non-empty output).
Goal: fix the 52% empty-relation negatives that made the model over-conservative and crashed recall.
Only the training set (train_e / train) is touched; valid/test are unchanged (empty-relation samples contribute no TP/FP/FN at eval).
"""
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data_end/data_generate"

# training files to filter (both text-only and entity-aware versions)
TARGETS = [
    BASE / "relation" / "relation_train.jsonl",
    BASE / "relation" / "relation_train_e.jsonl",
    BASE / "benchmark" / "relation_train.jsonl",
    BASE / "benchmark" / "relation_train_e.jsonl",
]

for f in TARGETS:
    if not f.exists():
        print(f"skip (not found): {f.name}")
        continue
    rows = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
    nonempty = [r for r in rows if r.get("output", [])]
    # back up the original file
    bak = f.with_suffix(".jsonl.bak_empty")
    if not bak.exists():
        shutil.copy(f, bak)
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in nonempty) + "\n", encoding="utf-8")
    print(f"{f.name}: {len(rows)}  rows → {len(nonempty)}  rows (removed {len(rows)-len(nonempty)}  rows empty relations)")
