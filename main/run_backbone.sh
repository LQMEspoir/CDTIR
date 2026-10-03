#!/usr/bin/env bash
# backbone comparison: run NER + RE on Qwen2.5-7B / GLM-4-9B-Chat / LLaMA-3.1-8B-Instruct, with args matching run_re_paper.sh.
#   all three models use the HF offline cache (~/.cache/huggingface/hub, symlinked to the data disk).
#   each model's artifacts live under outputs/backbone_<model>/.
#   Usage: bash run_backbone.sh                                   # run all three models
#         MODELS="glm4-9b" bash run_backbone.sh                  # run only one model
set -e
cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com

MODELS="${MODELS:-qwen2.5-7b,glm4-9b}"
EPOCHS_NER="${EPOCHS_NER:-4}"
EPOCHS_RE="${EPOCHS_RE:-4}"
DATA_DIR="${DATA_DIR:-data_end/data_merged}"
EMPTY_REL_WEIGHT="${EMPTY_REL_WEIGHT:-0.1}"

for model in ${MODELS//,/ }; do
  OUT="outputs/backbone_${model}"
  echo "==================== model $model ===================="

  echo "===== 1/6 train NER（$model, router_multilora, paper_ner）====="
  python run.py train \
    --task ner --split benchmark --data-dir "$DATA_DIR" \
    --model "$model" --strategy router_multilora \
    --prompt-profile paper_ner \
    --max-length 512 --epochs "$EPOCHS_NER" \
    --batch-size 2 --eval-batch-size 2 --grad-accum 2 \
    --lora-r 8 --lora-alpha 32 --lora-dropout 0.1 \
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.01 \
    --lr-scheduler-type cosine --max-grad-norm 1.0 \
    --router-hidden-mode frozen_base --router-loss-weight 0.5 \
    --eval-strategy steps --eval-steps 20 --save-strategy steps --save-steps 20 \
    --seed 2026 \
    --output-dir "$OUT/ner"

  echo "===== 2/6 predict NER ====="
  python run.py predict --task ner \
    --model "$model" \
    --router-dir "$OUT/ner" \
    --router-generation-mode weighted --router-generation-top-k 0 \
    --dataset-file "$DATA_DIR/benchmark/ner_test.jsonl" \
    --prompt-profile paper_ner \
    --output "$OUT/ner_predictions.jsonl"

  echo "===== 3/6 evaluate NER ====="
  python run.py evaluate --task ner \
    --predictions "$OUT/ner_predictions.jsonl" \
    --output-dir "$OUT/ner_eval"

  echo "===== 4/6 train RE（$model, balanced_lora, paper_re_e）====="
  python run.py train \
    --task relation --split benchmark --data-dir "$DATA_DIR" \
    --model "$model" --strategy balanced_lora \
    --prompt-profile paper_re_e \
    --max-length 1200 --epochs "$EPOCHS_RE" \
    --batch-size 2 --eval-batch-size 2 --grad-accum 2 \
    --lora-r 8 --lora-alpha 32 --lora-dropout 0.1 \
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.01 \
    --lr-scheduler-type cosine --max-grad-norm 1.0 \
    --eval-strategy steps --eval-steps 20 --save-strategy steps --save-steps 20 \
    --seed 2026 \
    --empty-relation-weight "$EMPTY_REL_WEIGHT" \
    --output-dir "$OUT/re"

  echo "===== 5/6 NER->RE predict ====="
  python run.py cdtir-predict --variant CDTIR \
    --model "$model" \
    --ner-router "$OUT/ner" \
    --re-adapter "$OUT/re/final_adapter" \
    --re-prompt-profile paper_re_e \
    --output-dir "$OUT/cdtir" \
    --max-new-tokens 256

  echo "===== 6/6 evaluate RE ====="
  python run.py evaluate --task relation \
    --predictions "$OUT/cdtir/re_e_pred_predictions.jsonl" \
    --output-dir "$OUT/re_eval"
  python run.py evaluate --task relation \
    --predictions "$OUT/cdtir/re_e_gold_predictions.jsonl" \
    --output-dir "$OUT/re_gold_eval"

  echo "==================== $model done: NER at $OUT/ner_eval/, RE at $OUT/re_eval/ / $OUT/re_gold_eval/ ===================="
done
