# CDTIR: Category-aware Dynamic Routing and Information Propagation for Tourist Intent Recognition

## Description

CDTIR (**C**ategory-aware **D**ynamic routing and information propagation for **T**ourist **I**ntent **R**ecognition) is a parameter-efficient framework that extracts structured travel-intent information (entities and relations) from user-generated travel-planning queries. It combines prompt engineering, large language models (LLMs), and LoRA fine-tuning to perform two subtasks:

- **Intent entity recognition** — 15 entity types grouped into four categories (*Location*, *Time*, *Service*, *Attribute*).
- **Relation extraction** — 6 relation types (*Play Order*, *Travel Time*, *Stay Duration*, *Budget Limit*, *Per-capita Budget*, *Experience*).

The framework has two core mechanisms: (1) a **category-aware dynamic routing mechanism** (MoRE, *Multi-LoRA with Router-based Entity Recognizer*) that adaptively fuses multiple LoRA branches according to the entity category, and (2) an **entity-aware information-propagation strategy** that passes NER predictions into the RE module as semantic priors.

The overall framework is illustrated below:

![CDTIR framework](docs/framework.png)

## Dataset

The corpus consists of user travel-planning queries crawled from three platforms (Ctrip Q&A community, Mafengwo Q&A, Xiaohongshu). Records are annotated via a multi-role LLM strategy followed by manual review.

- **Entities**: 15 types in 4 categories — Location (Destination/Origin/Return), Time (Time/Duration), Service (Dining/Lodging/Product/Transport), Attribute (Budget/Crowd/Intensity/Popularity/People/Weather).
- **Relations**: 6 types — Play Order, Travel Time, Stay Duration, Budget Limit, Per-capita Budget, Experience.
- **Size**: 2,726 records; train / valid / test = 2,211 / 258 / 257 (≈ 8:1:1). The 2,211 training records **already include** the 150 budget-class augmented samples (`aug_budget_*`), so the non-augmented corpus is 2,576 records.

| Split | Rows |
| --- | --- |
| train | 2,211 |
| valid | 258 |
| test | 257 |

> **Data availability.** The data that support the findings of this study are available from the corresponding author upon reasonable request.

### Data layout

All experiments use the **final dataset** at `data_end/data_merged/`, built from the original dataset (`data_origin`) plus 150 budget-augmented samples:

- `data_origin/` — the original dataset (flat: `entity_all.jsonl`, `relation_all.jsonl`, `relation_all_e.jsonl`; already collected and annotated, not yet augmented/cleaned/split).
- `data_aug/budget_samples.jsonl` — the 150 budget-augmented samples (`aug_budget_*`).
- `data_merged/` — the final dataset: `data_origin` + the 150 augmented samples appended, cleaned, then split 8:1:1 into train/valid/test.

Build `data_merged` (append → clean → split):

```bash
python src/build_data.py               # reuse the committed split (fixed test set)
python src/build_data.py --resplit     # force a fresh random 8:1:1 split
```

By default the split reuses the committed `split_membership.jsonl` (the test set stays fixed, so paper numbers stay reproducible). Pass `--resplit` to ignore it and re-split randomly.

**Augmentation note.** The 150 `aug_budget_*` samples are budget-class *entity* augmentation: each carries a `预算` (Budget) entity in NER and an **empty relation output (`[]`)** in RE (in both `relation_train.jsonl` and `relation_train_e.jsonl`). They participate in RE training as down-weighted "no-relation" examples — see `--empty-relation-weight` (0.1 in the entry scripts).

**Generation script.** These samples are produced by `scripts/gen_budget_aug.py` using a local Qwen2.5-7B (checked by `validate()` and de-duplicated, then written to `data_aug/budget_samples.jsonl`); `src/build_data.py` then merges them into `data_merged`, placing the augmented samples in the train split.

Data self-check:

```bash
python run.py smoke-test --verbose
```

## Code

```
main/
├── run.py                  # Unified CLI (train / predict / evaluate / benchmark / paper-*)
├── src/build_data.py      # Build data_merged from data_origin (append → clean → split)
├── src/tourism_ie/         # Core package
│   ├── training.py         # LoRA / MoRE (router_multilora) training
│   ├── inference.py        # Prediction (base model / adapter / router package)
│   ├── router_multilora.py # Category-aware dynamic routing (MoRE)
│   ├── router_grouped.py   # Grouped router (TF-IDF + LogisticRegression)
│   ├── paper.py            # Paper presets, data audit, NER→RE pipeline, reports (Table 9/10, Figure 4)
│   ├── modeling.py         # Model / tokenizer loading
│   ├── metrics.py          # Evaluation (micro/macro/weighted P/R/F1, accuracy)
│   ├── prompts.py          # Prompt templates (NER / RE profiles)
│   ├── data_utils.py       # Data paths, prompt rendering, entity formatting
│   ├── data_cleaning.py    # NER cleaning / correction (clean-data)
│   ├── data_split.py       # Benchmark splitting (split-data)
│   └── models/registry.py  # Backbone registry
├── scripts/                # Self-tests, manifests, data utilities
├── run_backbone.sh         # Multi-backbone comparison (default: qwen2.5-7b, glm4-9b)
├── backbone_compare.sh     # Alternative backbone comparison (qwen2.5-7b, glm4-9b, llama3.1-8b)
├── run_ablation.sh         # Ablation study (CDTIR / NMoRE / F / EH / TO)
├── paper_reproduce.sh      # End-to-end paper pipeline (ablation → Table 10 / Figure 4)
├── run_fewshot.sh          # Few-shot baseline (6 backbones)
├── run_zeroshot.sh         # Zero-shot baseline (6 backbones)
├── convert_baichuan_safetensors.py  # Baichuan2 .bin → safetensors converter
└── more/                   # Auxiliary scripts/tools unrelated to the paper experiments
```

## Usage

All commands run from `main/`; outputs are written to `outputs/`.

### Load the dataset

The merged dataset lives in `data_end/data_merged/` as JSONL:

- `ner/entity_all.jsonl` — entity annotations (`text_id`, `input`, `output: [{entity_text, entity_label}]`).
- `relation/relation_all.jsonl` — relation triples (`output: [{subject, predicate, object}]`).
- `relation/relation_all_e.jsonl` — entity-aware relation triples (adds the `entities` field).
- `benchmark/*.jsonl` — the train/valid/test splits.

Read a record with:

```python
import json
rows = [json.loads(line) for line in open("data_end/data_merged/ner/entity_all.jsonl", encoding="utf-8")]
```

### Entry scripts

#### Multi-backbone comparison

```bash
bash run_backbone.sh                     # default: qwen2.5-7b, glm4-9b
MODELS="qwen2.5-7b,glm4-9b,llama3.1-8b" bash run_backbone.sh   # add a third backbone
MODELS="qwen2.5-7b" bash run_backbone.sh # only one backbone
```

Each model's artifacts are under `outputs/backbone_<model>/` (`ner/`, `re/`, `cdtir/`, `ner_eval/`, `re_eval/`, `re_gold_eval/`). This produces the backbone comparison that feeds paper **Table 9**.

`backbone_compare.sh` is an alternative entry that runs the same NER + RE matrix through `run.py benchmark` (default backbones `qwen2.5-7b,glm4-9b,llama3.1-8b`).

#### Ablation study

```bash
bash run_ablation.sh      # CDTIR + 4 variants, produces Table 10
bash paper_reproduce.sh   # equivalent end-to-end pipeline (also runs paper-data audit + manifest)
```

| Variant | NER | RE |
| --- | --- | --- |
| CDTIR | MoRE (Router-MultiLoRA) | entity-aware LoRA (paper_re_e) |
| CDTIR_NMoRE | plain LoRA | reuse CDTIR |
| CDTIR_F | zero-shot | zero-shot |
| CDTIR_EH | reuse CDTIR | entity-aware hard (paper_re_eh) |
| CDTIR_TO | reuse CDTIR | text-only (paper_re_to) |

Artifacts are under `outputs/paper/<variant>/`; the ablation table is at `outputs/paper/reports/table_10.csv` (paper **Table 10**), together with `figure_4.{png,pdf,csv}` (paper **Figure 4**) and `re_e_gold_upper_bound.csv`.

#### Few-shot / zero-shot baselines

```bash
bash run_fewshot.sh    # 6 backbones: qwen3-8b, baichuan2-7b, qwen2.5-7b, glm4-9b, llama3.1-8b, mistral7b-v0.3
bash run_zeroshot.sh   # same 6 backbones, zero-shot
```

These scripts download one backbone at a time, run NER/RE, keep the outputs, then delete the weights to bound disk usage.

### Reproducing the paper results (minimal path)

1. **Install** the environment (`pip install -r requirements.txt`).
2. **Pre-download the base models** — the scripts run with `HF_HUB_OFFLINE=1` (see *Requirements → Models*).
3. **Audit the data**: `python run.py paper-data` (validates `data_end/data_merged`, writes `data_build_report.json` / `label_distribution.csv`).
4. **Backbone comparison (Table 9)**: `bash run_backbone.sh`.
5. **Ablation + Figure 4 (Table 10)**: `bash run_ablation.sh` (or `bash paper_reproduce.sh`).
6. **Baselines**: `bash run_fewshot.sh` and `bash run_zeroshot.sh`.

Key hyperparameters (from `src/tourism_ie/paper.py` and the entry scripts): LoRA r=8 / α=32 / dropout=0.1, batch 2 × grad-accum 2, lr 3e-5 (cosine, warmup-ratio 0.01), max-grad-norm 1.0, epochs 4, seed **2026**, router-loss-weight 0.5, empty-relation-weight 0.1; NER max-length 512, RE max-length 1200.

Outputs and their paper correspondence:

- `re_e_pred_predictions.jsonl` — the **real** NER→RE pipeline result (predicted entities as priors); this is the main reported relation number.
- `re_e_gold_predictions.jsonl` — the gold-entity upper bound, reported separately in `re_e_gold_upper_bound.csv`.
- `outputs/paper/reports/table_9.csv` / `table_10.csv` / `figure_4.*` — paper Table 9, Table 10, Figure 4.

> **Seed note.** The **training** seed is `2026` (unified across the entry scripts, `paper.py`, and `MANIFEST.json`). The **dataset split** uses a separate fixed seed (`42`, see `src/tourism_ie/data_split.py` and `split_membership.jsonl`); the two seeds are unrelated.

> **Hardware.** 7–9B backbones train with batch 2 / grad-accum 2; the reference runs used an RTX 5090 (24 GB). A single 24 GB+ GPU is recommended. If you hit OOM, add `--gradient-checkpointing` (training) or `--load-in-4bit` (inference).

### Common CLI

```bash
python run.py list-models          # list registered models
python run.py list-strategies      # list strategies
python run.py paper-data           # audit the paper dataset
python run.py paper-train --help   # train a paper preset (ablation)
python run.py cdtir-predict --help # NER → RE pipeline (Pred + Gold upper bound)
python run.py paper-report         # build Figure 4 / Table 9 / Table 10
python run.py train --help         # generic training
python run.py predict --help       # generic prediction
python run.py evaluate --task relation \
  --predictions outputs/cdtir_final/re_e_gold_predictions.jsonl \
  --output-dir outputs/re_gold_eval_final
```

Generic training example (NER MoRE):

```bash
python run.py train \
  --task ner --split benchmark --data-dir data_end/data_merged \
  --model qwen2.5-7b --strategy router_multilora \
  --prompt-profile paper_ner --max-length 512 --epochs 4 \
  --output-dir outputs/my_ner
```

## Requirements

- Python 3.12
- PyTorch 2.5.1+cu124 (CUDA 12.4) — match the CUDA build to your GPU driver (see the header of `requirements.txt`)
- transformers 4.57 (requirements.txt allows `>=4.48,<5`), peft, datasets, safetensors, openpyxl

```bash
pip install -r requirements.txt
```

### Models (HF offline cache)

Base models are loaded from the local Hugging Face cache (`~/.cache/huggingface/hub`, symlinked to `/root/autodl-tmp/hf_home/hub`):

| Registry key | HF repository | Note |
| --- | --- | --- |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B-Instruct | |
| `qwen3-8b` | Qwen/Qwen3-8B | |
| `glm4-9b` | zai-org/glm-4-9b-chat-hf | |
| `llama3.1-8b` | meta-llama/Meta-Llama-3.1-8B-Instruct | gated, accept license first |
| `mistral7b-v0.3` | mistralai/Mistral-7B-Instruct-v0.3 | |
| `baichuan2-7b` | baichuan-inc/Baichuan2-7B-Chat | `.bin` only; convert to safetensors first |

> **Offline default.** The entry scripts export `HF_HUB_OFFLINE=1`, so models are loaded **only** from the local cache. First-time users must download each base model (and accept gated licenses for LLaMA) before running. The few-shot/zero-shot scripts download one model at a time before switching to offline mode.

Download via mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-7B-Instruct
hf download zai-org/glm-4-9b-chat-hf
hf download meta-llama/Meta-Llama-3.1-8B-Instruct   # gated: run hf auth login first
```

> Note: Baichuan2-7B-Chat ships `pytorch_model.bin` (pickle) only; `torch.load(weights_only=True)` on torch<2.6 refuses it. Convert it to safetensors first (see `convert_baichuan_safetensors.py`).

> **Windows users.** The `.sh` entry scripts require Bash — use Git Bash or WSL.

## Method

CDTIR is backbone-agnostic and can be implemented on top of any supported instruction-tuned LLM (see the backbone registry in `src/tourism_ie/models/registry.py`). The ablation study uses **Qwen2.5-7B-Instruct** as the default backbone.

The pipeline has two stages:

1. **MoRE entity recognition (category-aware dynamic routing).** Given a backbone LLM, the router (a two-layer FNN with ReLU + Sigmoid) produces four per-category activation weights (*Location*, *Time*, *Service*, *Attribute*) from the mean-pooled last hidden state; the NER output is then generated by fusing the corresponding LoRA branches with those weights. Training uses a joint loss combining the text-generation loss and the router path-selection loss (balance coefficient 0.5).
2. **Entity-aware relation extraction.** NER predictions are passed into the RE module as entity priors together with head/tail type constraints, and relations are generated with a balanced LoRA objective.

## Citation

If you use this code or dataset, please cite:

> 基于类别感知动态路由与信息传播的游客意图识别 (Category-aware Dynamic Routing and Information Propagation for Tourist Intent Recognition).

## License & Contribution

This project is released for research and academic use. For other uses, please contact the authors. Contributions are welcome; please open an issue or pull request for discussion.

## Troubleshooting

- **CUDA OOM**: reduce `--batch-size`; training add `--gradient-checkpointing`; inference add `--load-in-4bit`.
- **RE-E missing entities**: entity-aware uses `relation_*_e.jsonl` (with `entities`); EH/TO use `relation_*.jsonl`.
- **Model load failure**: confirm `HF_HUB_OFFLINE=1` and the model is in the HF cache; LLaMA require license acceptance first.
