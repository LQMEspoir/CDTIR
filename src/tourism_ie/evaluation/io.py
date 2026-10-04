# File: src/tourism_ie/evaluation/io.py
# Module responsibility: evaluation result persistence utilities, writing overall metrics and per-category metrics as JSON/CSV.
# Main data flow: metrics dict -> create output directory -> JSON + per-class CSV.
# Reading suggestion: start with the public functions/classes in this file, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments, without modifying existing expressions, control flow, parameter defaults, or function call relationships.

from __future__ import annotations
import csv, json
from pathlib import Path


def save_metrics(metrics: dict, out_dir: Path, prefix: str = "metrics"):
    """[Function description] save_metrics
    - Responsibility: persists evaluation results as metrics.json and per-class CSV for benchmark aggregation and paper tables.
    - Main parameters: metrics: dict, out_dir: Path, prefix: str='metrics'.
    - Returns: returns the processed object/metrics/tensor/path, etc.; the actual structure depends on each return branch.
    - Note: may read/write disk files/model weights; ensure the output directory is writable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir/f"{prefix}.json"
    csv_path = out_dir/f"{prefix}_per_class.csv"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = metrics.get("per_class", [])
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return json_path, csv_path
