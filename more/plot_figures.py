#!/usr/bin/env python3
"""Generate all paper figures (data distribution + experiment results) in one script.

Usage: python plot_figures.py
Output: outputs/figures/*.png (300 dpi, paper figure format)

Figures:
  1. fig4_text_length_density   Text length distribution and density
  2. fig_entity_class_counts    Entity class frequency (15 classes)
  3. fig_relation_class_counts  Relation class frequency (6 classes)
  4. fig_relation_class_pie     Relation class distribution (donut)
  5. fig_entity_group_pie       Entity 4-group distribution (Location/Time/Service/Attribute)
  6. fig_entity_per_text        Entities per text distribution
  7. fig_ablation               Ablation study (5 variants NER/RE F1)
  8. fig_backbone_performance   Backbone comparison (NER/RE-E-Pred/RE-E-Gold F1)
  9. fig_training_loss          Training loss curves (qwen2.5-7b NER + RE)
"""
import json, glob, os, statistics
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import gaussian_kde

# ---- fonts (register CJK as fallback; primary is DejaVu Sans) ----
for _fp in ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    try:
        font_manager.fontManager.addfont(_fp)
    except Exception:
        pass
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---- Okabe-Ito colorblind-safe palette ----
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"
CATEGORY = [BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW]

# ---- Chinese label -> English ----
ENTITY_EN = {
    "目的地": "Destination", "出发地": "Origin", "返程地": "Return",
    "餐饮": "Dining", "住宿": "Lodging", "产品": "Product", "交通": "Transport",
    "预算": "Budget", "时长": "Duration", "时间": "Time",
    "人群": "Crowd", "强度": "Intensity", "人气": "Popularity", "人数": "People", "天气": "Weather",
}
REL_EN = {
    "游玩顺序": "Play Order", "出行时间": "Travel Time", "停留时长": "Stay Duration",
    "预算限制": "Budget Limit", "人均预算": "Per-capita Budget", "体验内容": "Experience",
}
ENTITY_GROUP = {
    "目的地": "Location", "出发地": "Location", "返程地": "Location",
    "时长": "Time", "时间": "Time",
    "餐饮": "Service", "住宿": "Service", "产品": "Service", "交通": "Service",
    "预算": "Attribute", "人群": "Attribute", "强度": "Attribute", "人气": "Attribute", "人数": "Attribute", "天气": "Attribute",
}

ROOT = str(Path(__file__).resolve().parent.parent)
OUT = f"{ROOT}/outputs/figures"
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)
    print("  ✓", name)


def _load_all_lens():
    # use entity_all (2726 unique text_id) to avoid train/test/e duplicate counting;
    # exclude budget-augmented synthetic samples (aug_budget_*) so the figure reflects real user queries
    lens = []
    for l in open(f"{ROOT}/data_end/data_merged/ner/entity_all.jsonl"):
        if l.strip():
            r = json.loads(l)
            if str(r.get("text_id", "")).startswith("aug_budget_"):
                continue
            lens.append(len(r.get("input", "")))
    return lens


def _load_ner_rows():
    return [json.loads(l) for l in open(f"{ROOT}/data_end/data_merged/ner/entity_all.jsonl") if l.strip()]


def _load_relation_rows():
    rows = []
    for split in ("train_e", "valid_e", "test_e"):
        for l in open(f"{ROOT}/data_end/data_merged/relation/relation_{split}.jsonl"):
            if l.strip():
                rows.append(json.loads(l))
    return rows


def fig_text_length():
    lens = _load_all_lens()
    lens = [x for x in lens if x <= 500]
    kde = gaussian_kde(lens, bw_method=1.5)
    xs = np.linspace(0, 500, 500)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(lens, bins=np.arange(0, 501, 50), density=True, alpha=0.35, color=BLUE, edgecolor="white", label="Histogram")
    ax.plot(xs, kde(xs), color=BLUE, linewidth=2.2, label="Density")
    hist_counts, _ = np.histogram(lens, bins=np.arange(0, 501, 50), density=True)
    ymax = max(hist_counts.max(), kde(xs).max()) * 1.25
    ax.axvline(statistics.mean(lens), color=ORANGE, linestyle="--", linewidth=1.4, label=f"Mean={statistics.mean(lens):.0f}")
    ax.axvline(statistics.median(lens), color=RED, linestyle="--", linewidth=1.4, label=f"Median={statistics.median(lens):.0f}")
    ax.set_xlabel("Text Length (chars)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Text Length Distribution", fontsize=13)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, 500)
    ax.set_xticks(np.arange(0, 501, 50))
    ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig4_text_length_density")


def fig_entity_class_counts():
    rows = _load_ner_rows()
    ent = Counter(e["entity_label"] for r in rows for e in r.get("output", []))
    labels = [ENTITY_EN.get(k, k) for k, v in ent.most_common()]
    vals = [v for k, v in ent.most_common()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(reversed(labels)), list(reversed(vals)), color=BLUE, height=0.7)
    for i, v in enumerate(reversed(vals)):
        ax.text(v + 15, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Entity Class Frequency", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_entity_class_counts")


def fig_relation_class_counts():
    rel = Counter(e.get("predicate") for r in _load_relation_rows() for e in r.get("output", []))
    labels = [REL_EN.get(k, k) for k, v in rel.most_common()]
    vals = [v for k, v in rel.most_common()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=CATEGORY[:len(labels)], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=9)
    ax.set_xlabel("Relation Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Relation Class Frequency", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_relation_class_counts")


def fig_relation_pie():
    rel = Counter(e.get("predicate") for r in _load_relation_rows() for e in r.get("output", []))
    labels = [REL_EN.get(k, k) for k, v in rel.most_common()]
    vals = [v for k, v in rel.most_common()]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    wedges, texts, autotexts = ax.pie(vals, autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "", startangle=90,
                                      colors=CATEGORY[:len(labels)], wedgeprops=dict(width=0.42, edgecolor="white"),
                                      pctdistance=0.78)
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
        t.set_color("white")
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)
    ax.set_title("Relation Class Distribution", fontsize=13)
    _save(fig, "fig_relation_class_pie")


def fig_entity_group_pie():
    """Merge 15 entity classes into 4 groups (Location/Time/Service/Attribute)."""
    rows = _load_ner_rows()
    ent = Counter(e["entity_label"] for r in rows for e in r.get("output", []))
    groups = {
        "Location": sum(ent[k] for k in ["目的地", "出发地", "返程地"]),
        # "Location": sum(ent[k] for k in ["Destination", "Origin", "Return"]),
        "Time": sum(ent[k] for k in ["时长", "时间"]),
        # "Time": sum(ent[k] for k in ["Duration", "Time"]),
        "Service": sum(ent[k] for k in ["餐饮", "住宿", "产品", "交通"]),
        # "Service": sum(ent[k] for k in ["Dining", "Lodging", "Product", "Transport"]),
        "Attribute": sum(ent[k] for k in ["预算", "人群", "强度", "人气", "人数", "天气"]),
        # "Attribute": sum(ent[k] for k in ["Budget", "Crowd", "Intensity", "Popularity", "People", "Weather"]),
    }
    labels = list(groups.keys())
    vals = list(groups.values())
    fig, ax = plt.subplots(figsize=(8.5, 6))
    wedges, texts, autotexts = ax.pie(vals, autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "", startangle=90,
                                      colors=CATEGORY[:len(labels)], wedgeprops=dict(width=0.42, edgecolor="white"),
                                      pctdistance=0.78)
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color("white")
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)
    ax.set_title("Entity Group Distribution", fontsize=13)
    _save(fig, "fig_entity_group_pie")


def fig_entity_per_text():
    rows = _load_ner_rows()
    c = Counter(len(r.get("output", [])) for r in rows)
    xs = sorted(c)
    ys = [c[x] for x in xs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, ys, color=BLUE, width=0.7)
    ax.set_xlabel("Entities per Text", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Entities per Text Distribution", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_entity_per_text")


def fig_ablation():
    """Ablation study: 5 variants x (NER, RE) micro-F1."""
    rows = {}
    for l in open(f"{ROOT}/outputs/paper/reports/table_10.csv", encoding="utf-8-sig"):
        p = l.strip().split(",")
        if len(p) < 5 or p[0] not in {"CDTIR", "CDTIR_NMoRE", "CDTIR_F", "CDTIR_EH", "CDTIR_TO"}:
            continue
        rows.setdefault(p[0], {})[p[1]] = float(p[4])  # micro_f1
    order = ["CDTIR", "CDTIR_NMoRE", "CDTIR_F", "CDTIR_EH", "CDTIR_TO"]
    names = ["CDTIR", "CDTIR_NMoRE", "CDTIR_F", "CDTIR_EH", "CDTIR_TO"]
    ner = [rows[v].get("ner", 0) for v in order]
    rel = [rows[v].get("relation", 0) for v in order]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - w / 2, ner, w, label="NER F1", color=BLUE)
    b2 = ax.bar(x + w / 2, rel, w, label="RE F1", color=ORANGE)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("micro F1", fontsize=11)
    ax.set_title("Ablation Study", fontsize=13)
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_ablation")


def fig_backbone_performance():
    models = ["qwen3-8b", "qwen2.5-7b", "glm4-9b", "baichuan2-7b", "llama3.1-8b", "mistral7b-v0.3"]
    names = ["Qwen3-8B", "Qwen2.5-7B", "GLM-4-9B", "Baichuan2-7B", "LLaMA-3.1-8B", "Mistral-7B"]
    ner, pred, gold = [], [], []
    for m in models:
        nf = sorted(glob.glob(f"{ROOT}/outputs/backbone_{m}/ner_eval/*metrics.json"))
        pf = sorted(glob.glob(f"{ROOT}/outputs/backbone_{m}/re_eval/*metrics.json"))
        gf = sorted(glob.glob(f"{ROOT}/outputs/backbone_{m}/re_gold_eval/*metrics.json"))
        ner.append(json.load(open(nf[0]))["micro"]["f1"])
        pred.append(json.load(open(pf[0]))["micro"]["f1"] if pf else 0)
        gold.append(json.load(open(gf[0]))["micro"]["f1"] if gf else 0)
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w, ner, w, label="NER F1", color=BLUE)
    b2 = ax.bar(x, pred, w, label="RE-E-Pred F1", color=ORANGE)
    b3 = ax.bar(x + w, gold, w, label="RE-E-Gold F1", color=GREEN)
    for b in list(b1) + list(b2) + list(b3):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("micro F1", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13)
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_backbone_performance")


def fig_training_loss():
    """qwen2.5-7b NER + RE training loss curves (eval_loss vs step)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for task, color, label in [("ner", BLUE, "NER"), ("re", ORANGE, "RE")]:
        d = json.load(open(f"{ROOT}/outputs/backbone_qwen2.5-7b/{task}/training_report.json"))
        h = d.get("eval_history", [])
        if h:
            ax.plot([x["step"] for x in h], [x["eval_loss"] for x in h], color=color, linewidth=2, label=label)
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.set_title("Training Loss (Qwen2.5-7B)", fontsize=13)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_training_loss")


def fig_three_settings():
    """Fine-tuned / few-shot / zero-shot x 6 backbones (NER + RE-E-Pred + RE-E-Gold)."""
    models = ["qwen3-8b", "qwen2.5-7b", "glm4-9b", "baichuan2-7b", "llama3.1-8b", "mistral7b-v0.3"]
    names = ["Qwen3-8B", "Qwen2.5-7B", "GLM-4-9B", "Baichuan2-7B", "LLaMA-3.1-8B", "Mistral-7B"]

    def f1(prefix, key, task):
        sub = {"ner": "ner_eval", "re_pred": "re_eval", "re_gold": "re_gold_eval"}[task]
        fs = sorted(glob.glob(f"{ROOT}/outputs/{prefix}{key}/{sub}/*metrics.json"))
        return json.load(open(fs[0]))["micro"]["f1"] if fs else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, task, title in [(axes[0], "ner", "NER"), (axes[1], "re_pred", "RE-Pred"), (axes[2], "re_gold", "RE-Gold")]:
        x = np.arange(len(names))
        for color, label, prefix in [(BLUE, "CDTIR", "backbone_"), (ORANGE, "Few-Shot", "few-shot/"), (RED, "Zero-Shot", "zero-shot/")]:
            y = [f1(prefix, m, task) for m in models]
            ax.plot(x, y, marker="o", markersize=5, linewidth=2, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("micro F1", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(frameon=False)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model Performance across Settings", fontsize=13)
    _save(fig, "fig_three_settings")


def fig_per_class_f1():
    """NER 15-class + RE 6-class per-class F1 (Qwen2.5-7B)."""
    ner = json.load(open(glob.glob(f"{ROOT}/outputs/backbone_qwen2.5-7b/ner_eval/*metrics.json")[0]))
    rel = json.load(open(glob.glob(f"{ROOT}/outputs/backbone_qwen2.5-7b/re_gold_eval/*metrics.json")[0]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, data, title, color, mapping in [
        (axes[0], ner, "NER Per-Class F1", BLUE, ENTITY_EN),
        (axes[1], rel, "RE Per-Class F1", ORANGE, REL_EN),
    ]:
        pc = sorted(data["per_class"], key=lambda c: -c["f1"])
        labels = [mapping.get(c["label"], c["label"]) for c in pc]
        vals = [c["f1"] for c in pc]
        ax.barh(list(reversed(labels)), list(reversed(vals)), color=color, height=0.65)
        for i, v in enumerate(reversed(vals)):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
        ax.set_xlabel("F1", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per-Class Performance (Qwen2.5-7B)", fontsize=13)
    _save(fig, "fig_per_class_f1")


def fig_longtail():
    """F1 vs support scatter, showing the long-tail effect."""
    ner = json.load(open(glob.glob(f"{ROOT}/outputs/backbone_qwen2.5-7b/ner_eval/*metrics.json")[0]))
    rel = json.load(open(glob.glob(f"{ROOT}/outputs/backbone_qwen2.5-7b/re_gold_eval/*metrics.json")[0]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    points = []  # (support, f1, label, color)
    for data, color, label, mapping in [(ner, BLUE, "NER", ENTITY_EN), (rel, ORANGE, "RE", REL_EN)]:
        pts = [c for c in data["per_class"] if c["support"] > 0]
        ax.scatter([c["support"] for c in pts], [c["f1"] for c in pts], color=color, s=45, alpha=0.7, label=label)
        for c in pts:
            points.append((c["support"], c["f1"], mapping.get(c["label"], c["label"]), color))
    # head classes (support>10) are annotated directly; long-tail classes (support<=10) use leader lines to the right label column
    head = [p for p in points if p[0] > 10]
    tail = sorted([p for p in points if p[0] <= 10], key=lambda p: (p[0], p[1]))
    for s, f, lab, color in head:
        ax.annotate(lab, (s, f), textcoords="offset points", xytext=(6, 3), fontsize=7, color=color)
    for k, (s, f, lab, color) in enumerate(tail):
        y = 0.95 - k * 0.13
        ax.annotate(lab, xy=(s, f), xytext=(0.83, y), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.6),
                    fontsize=7, color=color, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Support (log scale)", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("Performance vs Support (Long-tail)", fontsize=13)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_longtail")


def _confusion_data(predictions_path, task):
    """Build class-to-class confusion data (+<Missed>/<False>) for NER or RE."""
    from collections import defaultdict, Counter
    rows = [json.loads(l) for l in open(predictions_path) if l.strip()]
    conf = defaultdict(int)
    if task == "ner":
        mapping = ENTITY_EN
        for r in rows:
            gold = {e["entity_text"]: e["entity_label"] for e in r.get("gold", [])}
            pred = {e["entity_text"]: e["entity_label"] for e in r.get("prediction", [])}
            for txt, gl in gold.items():
                conf[(gl, pred.get(txt, "<Missed>"))] += 1
            for txt, pl in pred.items():
                if txt not in gold:
                    conf[("<False>", pl)] += 1
    else:  # relation
        mapping = REL_EN
        for r in rows:
            gold = {(e.get("subject"), e.get("object")): e.get("predicate") for e in r.get("gold", [])}
            pred = {(e.get("subject"), e.get("object")): e.get("predicate") for e in r.get("prediction", [])}
            for k, gl in gold.items():
                conf[(gl, pred.get(k, "<Missed>"))] += 1
            for k, pl in pred.items():
                if k not in gold:
                    conf[("<False>", pl)] += 1
    freq = Counter(gl for (gl, _), _ in conf.items() if gl != "<False>")
    top = [l for l, _ in freq.most_common(8)]
    labels = [mapping.get(l, l) for l in top] + ["<False>"]
    cols = [mapping.get(l, l) for l in top] + ["<Missed>"]
    raw_labels = top + ["<False>"]
    raw_cols = top + ["<Missed>"]
    n = len(labels)
    mat = np.zeros((n, n))
    for i, gl in enumerate(raw_labels):
        for j, pl in enumerate(raw_cols):
            mat[i][j] = conf.get((gl, pl), 0)
    return labels, cols, mat


def _draw_confusion(ax, labels, cols, mat, title):
    n = len(labels)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    for i in range(n):
        for j in range(n):
            v = int(mat[i][j])
            color = "white" if mat[i][j] > mat.max() * 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color=color, fontweight="bold")
    return im


def fig_confusion_matrix():
    """NER confusion matrix (top 8 classes + missed/false)."""
    labels, cols, mat = _confusion_data(f"{ROOT}/outputs/backbone_qwen2.5-7b/ner_predictions.jsonl", "ner")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = _draw_confusion(ax, labels, cols, mat, "NER Confusion Matrix (Qwen2.5-7B, top 8)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "fig_confusion_matrix")


def fig_re_confusion_matrix():
    """RE confusion matrices (RE-E-Pred + RE-E-Gold)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, path, title in [
        (axes[0], f"{ROOT}/outputs/backbone_qwen2.5-7b/cdtir/re_e_pred_predictions.jsonl", "RE-E-Pred"),
        (axes[1], f"{ROOT}/outputs/backbone_qwen2.5-7b/cdtir/re_e_gold_predictions.jsonl", "RE-E-Gold"),
    ]:
        labels, cols, mat = _confusion_data(path, "relation")
        im = _draw_confusion(ax, labels, cols, mat, title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("RE Confusion Matrix (Qwen2.5-7B)", fontsize=13)
    _save(fig, "fig_re_confusion_matrix")


def fig_parameter_efficiency():
    """Trainable LoRA parameters per model (trainable ratio <0.5%)."""
    models = ["qwen3-8b", "qwen2.5-7b", "glm4-9b", "baichuan2-7b", "llama3.1-8b", "mistral7b-v0.3"]
    names = ["Qwen3-8B", "Qwen2.5-7B", "GLM-4-9B", "Baichuan2-7B", "LLaMA-3.1-8B", "Mistral-7B"]
    trainable, total = [], []
    for m in models:
        d = json.load(open(f"{ROOT}/outputs/backbone_{m}/ner/training_report.json"))
        trainable.append(d["trainable_parameters"] / 1e6)
        total.append(d["total_parameters"] / 1e9)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(names, trainable, color=BLUE, width=0.6)
    for b, v, t in zip(bars, trainable, total):
        pct = v / (t * 1000) * 100
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}M\n({pct:.2f}%)", ha="center", fontsize=8)
    ax.set_ylabel("Trainable Parameters (M)", fontsize=11)
    ax.set_title("LoRA Parameter Efficiency (<0.5% trainable)", fontsize=13)
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_parameter_efficiency")


def fig_experiment_env():
    """Experiment environment table."""
    items = [
        "GPU: NVIDIA vGPU-32GB (VRAM 32 GB)",
        "CPU: 12 vCPU Intel(R) Xeon(R) Platinum 8352V",
        "Memory: 40 GB",
        "OS: Ubuntu 20.04",
        "CUDA: 12.4",
        "Framework: PyTorch 2.6.0",
    ]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axis("off")
    for i, line in enumerate(items):
        ax.text(0.05, 0.82 - i * 0.13, line, fontsize=11, transform=ax.transAxes)
    ax.set_title("Experiment Environment", fontsize=13)
    _save(fig, "fig_experiment_env")


def fig_router_analysis():
    """Router confidence, routed-group distribution, and routing accuracy (Qwen2.5-7B)."""
    rows = [json.loads(l) for l in open(f"{ROOT}/outputs/backbone_qwen2.5-7b/ner_predictions.jsonl") if l.strip()]
    GROUP_KEY = {"location": "Location", "time": "Time", "service": "Service", "attribute": "Attribute"}
    confs = [r.get("router_confidence", 0.0) for r in rows]
    # routing accuracy: routed group vs dominant gold group
    correct = total = 0
    for r in rows:
        gc = Counter(ENTITY_GROUP[e["entity_label"]] for e in r.get("gold", []) if e.get("entity_label") in ENTITY_GROUP)
        if gc:
            true_group = gc.most_common(1)[0][0].lower()
            if r.get("router_adapter") == true_group:
                correct += 1
            total += 1
    acc = correct / total if total else 0.0
    group_counts = Counter(r.get("router_adapter") for r in rows)
    order = ["location", "time", "service", "attribute"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(confs, bins=20, color=BLUE, alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("Router Confidence"); axes[0].set_ylabel("Count")
    axes[0].set_title("Router Confidence Distribution")
    axes[1].bar([GROUP_KEY[g] for g in order], [group_counts.get(g, 0) for g in order], color=CATEGORY[:4])
    axes[1].set_xlabel("Routed Group"); axes[1].set_ylabel("Count")
    axes[1].set_title("Router Group Distribution")
    axes[2].text(0.5, 0.62, f"{acc:.1%}", ha="center", fontsize=42, color=BLUE, fontweight="bold")
    axes[2].text(0.5, 0.38, "Routing Accuracy\n(vs dominant gold group)", ha="center", fontsize=11)
    axes[2].axis("off"); axes[2].set_title("Routing Accuracy")
    for a in axes[:2]:
        a.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_router_analysis")


def fig_re_ablation():
    """RE ablation: prompt (paper_re_e/v3) x strategy (lora/balanced_lora), Qwen2.5-7B."""
    def f1(base, sub):
        fs = glob.glob(f"{base}/{sub}/*metrics.json")
        return json.load(open(fs[0]))["micro"]["f1"] if fs else 0.0
    configs = [
        ("lora", "paper_re_e", "outputs/lora_baseline/qwen2.5-7b"),
        ("lora", "paper_re_e_v3", "outputs/RE/qwen2.5-7b"),
        ("balanced_lora", "paper_re_e", "outputs/RE2/qwen2.5-7b"),
        ("balanced_lora", "paper_re_e_v3", "outputs/backbone_qwen2.5-7b"),
    ]
    labels = [f"{'Bal-LoRA' if s == 'balanced_lora' else 'LoRA'}\n{'few-shot' if p == 'paper_re_e_v3' else 'no few-shot'}" for s, p, _ in configs]
    pred = [f1(b, "re_eval") for _, _, b in configs]
    gold = [f1(b, "re_gold_eval") for _, _, b in configs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, vals, title in [(axes[0], pred, "RE-E-Pred F1"), (axes[1], gold, "RE-E-Gold F1")]:
        bars = ax.bar(labels, vals, color=[BLUE, SKY, ORANGE, RED], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylabel("micro F1", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("RE Ablation: Prompt x Strategy (Qwen2.5-7B)", fontsize=13)
    _save(fig, "fig_re_ablation")


def fig_error_propagation():
    """NER error propagation: RE-E-Pred vs RE-E-Gold gap per backbone."""
    models = ["qwen3-8b", "qwen2.5-7b", "glm4-9b", "baichuan2-7b", "llama3.1-8b", "mistral7b-v0.3"]
    names = ["Qwen3-8B", "Qwen2.5-7B", "GLM-4-9B", "Baichuan2-7B", "LLaMA-3.1-8B", "Mistral-7B"]
    pred, gold = [], []
    for m in models:
        pf = glob.glob(f"{ROOT}/outputs/backbone_{m}/re_eval/*metrics.json")
        gf = glob.glob(f"{ROOT}/outputs/backbone_{m}/re_gold_eval/*metrics.json")
        pred.append(json.load(open(pf[0]))["micro"]["f1"])
        gold.append(json.load(open(gf[0]))["micro"]["f1"] if gf else 0.0)
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - w / 2, pred, w, label="RE-E-Pred (predicted entities)", color=ORANGE)
    b2 = ax.bar(x + w / 2, gold, w, label="RE-E-Gold (gold entities)", color=GREEN)
    for i in range(len(names)):
        gap = gold[i] - pred[i]
        ax.text(x[i], max(pred[i], gold[i]) + 0.02, f"Δ{abs(gap):.2f}", ha="center", fontsize=8, color=RED)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("micro F1", fontsize=11)
    ax.set_title("NER Error Propagation to RE", fontsize=13)
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_error_propagation")


def fig_entity_group_perf():
    """Entity 4-group NER F1 (Qwen2.5-7B): aggregate per-class F1 to groups."""
    ner = json.load(open(glob.glob(f"{ROOT}/outputs/backbone_qwen2.5-7b/ner_eval/*metrics.json")[0]))
    groups = {"Location": [], "Time": [], "Service": [], "Attribute": []}
    for c in ner["per_class"]:
        g = ENTITY_GROUP.get(c["label"])
        if g:
            groups[g].append(c["f1"])
    labels = list(groups.keys())
    vals = [sum(v) / len(v) if v else 0.0 for v in groups.values()]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, color=CATEGORY[:len(labels)], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("macro F1", fontsize=11)
    ax.set_title("NER F1 by Entity Group (Qwen2.5-7B)", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_entity_group_perf")


if __name__ == "__main__":
    print("Generating figures to", OUT)
    fig_text_length()
    fig_entity_class_counts()
    fig_relation_class_counts()
    fig_relation_pie()
    fig_entity_group_pie()
    fig_entity_per_text()
    fig_ablation()
    fig_backbone_performance()
    fig_training_loss()
    fig_three_settings()
    fig_per_class_f1()
    fig_longtail()
    fig_confusion_matrix()
    fig_re_confusion_matrix()
    fig_parameter_efficiency()
    fig_experiment_env()
    fig_router_analysis()
    fig_re_ablation()
    fig_error_propagation()
    fig_entity_group_perf()
    print("Done,", len(os.listdir(OUT)), "figures")
