#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen2.5-7b}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CDTIR_ROOT:-$PWD}"
export CDTIR_ROOT="$WORK_DIR"
export CDTIR_MODEL_ROOT="${CDTIR_MODEL_ROOT:-../models}"
RUN_PY="$SCRIPT_DIR/run.py"
PAPER_OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-outputs/paper}"
export CDTIR_PAPER_OUTPUT_ROOT="$PAPER_OUTPUT_ROOT"

python "$RUN_PY" paper-data
python "$RUN_PY" paper-train --variant CDTIR --model "$MODEL" --max-steps "$MAX_STEPS"
python "$RUN_PY" paper-train --variant CDTIR_NMoRE --model "$MODEL" --max-steps "$MAX_STEPS"
python "$RUN_PY" paper-train --variant CDTIR_EH --task relation --model "$MODEL" --max-steps "$MAX_STEPS"
python "$RUN_PY" paper-train --variant CDTIR_TO --task relation --model "$MODEL" --max-steps "$MAX_STEPS"
python "$RUN_PY" paper-train --variant CDTIR_F --model "$MODEL"

python "$RUN_PY" cdtir-predict --variant CDTIR --model "$MODEL" \
  --ner-router "$PAPER_OUTPUT_ROOT/CDTIR/ner" --re-adapter "$PAPER_OUTPUT_ROOT/CDTIR/relation/final_adapter" \
  --max-new-tokens "$MAX_NEW_TOKENS"
python "$RUN_PY" cdtir-predict --variant CDTIR_NMoRE --model "$MODEL" \
  --ner-adapter "$PAPER_OUTPUT_ROOT/CDTIR_NMoRE/ner/final_adapter" \
  --re-adapter "$PAPER_OUTPUT_ROOT/CDTIR/relation/final_adapter" \
  --max-new-tokens "$MAX_NEW_TOKENS"
python "$RUN_PY" cdtir-predict --variant CDTIR_F --model "$MODEL" \
  --max-new-tokens "$MAX_NEW_TOKENS"

mkdir -p "$PAPER_OUTPUT_ROOT"/{CDTIR,CDTIR_NMoRE,CDTIR_EH,CDTIR_TO,CDTIR_F}/predictions
cp "$PAPER_OUTPUT_ROOT/CDTIR/pipeline/ner_predictions.jsonl" "$PAPER_OUTPUT_ROOT/CDTIR/predictions/ner.jsonl"
cp "$PAPER_OUTPUT_ROOT/CDTIR/pipeline/re_e_pred_predictions.jsonl" "$PAPER_OUTPUT_ROOT/CDTIR/predictions/relation.jsonl"
cp "$PAPER_OUTPUT_ROOT/CDTIR_NMoRE/pipeline/ner_predictions.jsonl" "$PAPER_OUTPUT_ROOT/CDTIR_NMoRE/predictions/ner.jsonl"
cp "$PAPER_OUTPUT_ROOT/CDTIR_NMoRE/pipeline/re_e_pred_predictions.jsonl" "$PAPER_OUTPUT_ROOT/CDTIR_NMoRE/predictions/relation.jsonl"
cp "$PAPER_OUTPUT_ROOT/CDTIR_F/pipeline/ner_predictions.jsonl" "$PAPER_OUTPUT_ROOT/CDTIR_F/predictions/ner.jsonl"
cp "$PAPER_OUTPUT_ROOT/CDTIR_F/pipeline/re_e_pred_predictions.jsonl" "$PAPER_OUTPUT_ROOT/CDTIR_F/predictions/relation.jsonl"
cp "$PAPER_OUTPUT_ROOT/CDTIR/predictions/ner.jsonl" "$PAPER_OUTPUT_ROOT/CDTIR_EH/predictions/ner.jsonl"
cp "$PAPER_OUTPUT_ROOT/CDTIR/predictions/ner.jsonl" "$PAPER_OUTPUT_ROOT/CDTIR_TO/predictions/ner.jsonl"

python "$RUN_PY" predict --task relation --model "$MODEL" --adapter "$PAPER_OUTPUT_ROOT/CDTIR_EH/relation/final_adapter" \
  --dataset-file data_end/data_merged/relation/relation_test.jsonl --prompt-profile paper_re_eh \
  --output "$PAPER_OUTPUT_ROOT/CDTIR_EH/predictions/relation.jsonl" --max-new-tokens "$MAX_NEW_TOKENS"
python "$RUN_PY" predict --task relation --model "$MODEL" --adapter "$PAPER_OUTPUT_ROOT/CDTIR_TO/relation/final_adapter" \
  --dataset-file data_end/data_merged/relation/relation_test.jsonl --prompt-profile paper_re_to \
  --output "$PAPER_OUTPUT_ROOT/CDTIR_TO/predictions/relation.jsonl" --max-new-tokens "$MAX_NEW_TOKENS"

python "$RUN_PY" paper-report --output-dir "$PAPER_OUTPUT_ROOT/reports"
python "$SCRIPT_DIR/scripts/build_manifest.py"
