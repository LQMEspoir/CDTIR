# File: src/tourism_ie/metrics.py
# Module responsibility: a lightweight wrapper over the new evaluation.metrics; loads prediction files and calls the detailed metric implementation.
# Main data flow: prediction JSONL -> load gold/pred -> detailed evaluator -> metrics file.
# Reading guide: start with this file's public functions/classes, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments; it does not modify existing expressions, control flow, default parameter values, or function call relationships.

from __future__ import annotations
from pathlib import Path
import json
from .evaluation import evaluate_records_detailed, save_metrics


def evaluate_records(task,records):
    """[Function description] evaluate_records
    - Responsibility: calls the new detailed evaluator on in-memory prediction records and returns overall and per-category metrics.
    - Main parameters: task, records.
    - Returns: returns processed objects/metrics/tensors/paths, etc.; the actual structure is decided by each return branch.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the called objects, it has no extra persistence side effects.
    """
    """Backward-compatible summary plus detailed metrics."""
    d=evaluate_records_detailed(task,records)
    return {
        "samples":d["samples"],"tp":d["micro"]["tp"],"fp":d["micro"]["fp"],"fn":d["micro"]["fn"],
        "precision":d["micro"]["precision"],"recall":d["micro"]["recall"],"f1":d["micro"]["f1"],
        "macro_f1":d["macro"]["f1"],"weighted_f1":d["weighted"]["f1"],
        "accuracy":d["accuracy"],
        "exact_sample_accuracy":d["exact_sample_accuracy"],"per_class":d["per_class"],
    }


def load_prediction_records(path: Path):
    """[Function description] load_prediction_records
    - Responsibility: loads previously saved prediction records from a JSONL file.
    - Main parameters: path: Path.
    - Returns: returns processed objects/metrics/tensors/paths, etc.; the actual structure is decided by each return branch.
    - Note: may read/write disk files/model weights; please ensure the output directory is writable.
    """
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def evaluate_file(root: Path,task:str,path:Path,output_dir:Path|None=None,legacy_text_metrics:bool=False,plot_metrics:bool=False):
    """[Function description] evaluate_file
    - Responsibility: reads the prediction file, runs the unified evaluation, and saves the JSON/CSV metrics to the output directory.
    - Main parameters: root: Path, task: str, path: Path, output_dir: Path | None=None, legacy_text_metrics: bool=False, plot_metrics: bool=False.
    - Returns: returns processed objects/metrics/tensors/paths, etc.; the actual structure is decided by each return branch.
    - Note: may read/write disk files/model weights; please ensure the output directory is writable; may print log or warning messages.
    """
    rows=load_prediction_records(path)
    m=evaluate_records_detailed(task,rows)
    if legacy_text_metrics:
        from .legacy_metrics import compute_legacy_text_metrics
        m["legacy_text_metrics"]=compute_legacy_text_metrics(task,rows)
    out=output_dir or path.parent
    json_path,csv_path=save_metrics(m,out,prefix=f"{path.stem}_metrics")
    plot_path=None
    if plot_metrics:
        from .legacy_metrics import plot_metrics as _plot
        plot_path=_plot(m,out/"all_metrics.png")
    print(json.dumps(m,ensure_ascii=False,indent=2))
    print(f"Detailed metrics JSON: {json_path}")
    print(f"Per-class metrics CSV: {csv_path}")
    if plot_path: print(f"Metric plot: {plot_path}")
    return m
