#!/usr/bin/env bash
set -euo pipefail

# Run the requested backbone comparison over the same paper split:
#   NER: Router-MultiLoRA
#   RE:  entity-aware LoRA (paper_re_e by default)
# The default is a one-epoch smoke run: 809 rows, batch=2, accumulation=2,
# hence 203 optimizer steps. Use MAX_STEPS=-1 and EPOCHS=4 for final results.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep outputs in the caller's current directory.  The script path is used
# only to locate run.py; CDTIR_ROOT is an explicit opt-in override.
WORK_DIR="${CDTIR_ROOT:-$PWD}"
export CDTIR_ROOT="$WORK_DIR"
MODEL_ROOT="${MODEL_ROOT:-../models}"
export CDTIR_MODEL_ROOT="${CDTIR_MODEL_ROOT:-$MODEL_ROOT}"
RUN_PY="$SCRIPT_DIR/run.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-qwen2.5-7b,glm4-9b,llama3.1-8b}"
DATA_DIR="${DATA_DIR:-data_end/data_merged}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/backbone_compare}"
RE_PROMPT_PROFILE="${RE_PROMPT_PROFILE:-paper_re_e}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-203}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
DRY_RUN="${DRY_RUN:-0}"

DRY_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  DRY_ARGS+=(--dry-run)
fi

"$PYTHON_BIN" "$RUN_PY" paper-data --data-dir "$DATA_DIR"

COMMON=(
  --models "$MODELS"
  --data-dir "$DATA_DIR"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --learning-rate 3e-5
  --weight-decay 0.01
  --warmup-ratio 0.01
  --lr-scheduler-type cosine
  --lora-r 8
  --lora-alpha 32
  --lora-dropout 0.1
  --target-format original
  --logging-steps 10
  --eval-strategy steps
  --eval-steps 20
  --save-strategy steps
  --save-steps 20
  --save-total-limit 2
  --max-grad-norm 1.0
  --max-new-tokens "$MAX_NEW_TOKENS"
  --fail-fast
)

"$PYTHON_BIN" "$RUN_PY" benchmark --task ner \
  --strategies router_multilora \
  --prompt-profile paper_ner \
  --max-length 512 \
  --router-hidden-mode frozen_base \
  --router-loss-weight 0.5 \
  --router-generation-mode weighted \
  "${COMMON[@]}" \
  "${DRY_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR/ner"

"$PYTHON_BIN" "$RUN_PY" benchmark --task relation \
  --strategies lora \
  --prompt-profile "$RE_PROMPT_PROFILE" \
  --max-length 1200 \
  "${COMMON[@]}" \
  "${DRY_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR/relation"

echo "Backbone comparison finished. Results: $OUTPUT_DIR"
