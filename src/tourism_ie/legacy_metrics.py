# File: src/tourism_ie/legacy_metrics.py
# Module responsibility: legacy text-generation metrics (BLEU/ROUGE) and simple chart output for old papers/scripts.
# Main data flow: predicted vs reference text -> text alignment -> BLEU/ROUGE -> metric dict/chart.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
import json
from pathlib import Path


def _text_pairs(task: str, records: list[dict]):
    """- Responsibility: extracts reference/prediction text pairs from prediction records for BLEU/ROUGE.
    - Parameters: task: str,records: list[dict].
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    preds=[]; refs=[]
    for r in records:
        if "gold" not in r: continue
        pred=str(r.get("raw_prediction") or json.dumps(r.get("prediction",[]),ensure_ascii=False,separators=(",",":")))
        ref=json.dumps(r.get("gold",[]),ensure_ascii=False,separators=(",",":"))
        preds.append(pred); refs.append(ref)
    return preds,refs


def compute_legacy_text_metrics(task: str, records: list[dict]) -> dict:
    """- Responsibility: computes legacy-compatible text-generation metrics and returns a report-ready dict.
    - Parameters: task: str,records: list[dict].
    - Returns: return type is `dict`; see implementation below for field details.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    preds,refs=_text_pairs(task,records)
    if not preds:
        return {"bleu4":0.0,"rougeL":0.0,"text_metric_samples":0}
    try:
        import sacrebleu
    except ImportError as e:
        raise SystemExit("BLEU-4 requested. Install: pip install -r requirements.txt") from e
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise SystemExit("ROUGE-L requested. Install: pip install -r requirements.txt") from e
    bleu=float(sacrebleu.corpus_bleu(preds,[refs],smooth_method="exp").score)
    scorer=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=False)
    rouge=sum(scorer.score(ref,pred)["rougeL"].fmeasure for pred,ref in zip(preds,refs))/len(preds)*100.0
    return {"bleu4":bleu,"rougeL":float(rouge),"text_metric_samples":len(preds)}


def plot_metrics(metrics: dict, out_path: Path):
    """- Responsibility: plots legacy metrics or training-process data into an image file.
    - Parameters: metrics: dict,out_path: Path.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;includes explicit parameter/data validation; raises on invalid input.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Metric plot requested. Install: pip install -r requirements.txt") from e
    micro=metrics.get("micro",{})
    names=[]; values=[]
    if "legacy_text_metrics" in metrics:
        names += ["BLEU-4","ROUGE-L"]
        values += [metrics["legacy_text_metrics"].get("bleu4",0.0),metrics["legacy_text_metrics"].get("rougeL",0.0)]
    names += ["Precision","Recall","F1"]
    values += [100*micro.get("precision",0.0),100*micro.get("recall",0.0),100*micro.get("f1",0.0)]
    plt.rcParams["axes.unicode_minus"]=False
    fig=plt.figure(figsize=(10,5)); ax=fig.add_subplot(111)
    bars=ax.bar(names,values); ax.set_ylim(0,100); ax.set_title("Model evaluation metrics")
    for bar,v in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,v+1,f"{v:.2f}",ha="center")
    fig.tight_layout(); fig.savefig(out_path,dpi=180); plt.close(fig)
    return out_path
