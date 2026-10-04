#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CDTIR_ROOT:-$PWD}"
export CDTIR_ROOT="$WORK_DIR"
export CDTIR_MODEL_ROOT="${CDTIR_MODEL_ROOT:-../models}"
RUN_PY="$SCRIPT_DIR/run.py"
MODEL_ROOT="${MODEL_ROOT:-../models}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/Qwen2.5-0.5B-Instruct}"
MODEL_KEY="${MODEL_KEY:-qwen2.5-0.5b}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
python -m pip install -r "$SCRIPT_DIR/requirements.txt"
if [[ -f "$MODEL_DIR/config.json" ]]; then MODEL_ARG="$MODEL_DIR"; else MODEL_ARG="$MODEL_KEY"; fi
python "$SCRIPT_DIR/scripts/materialize_reviewed_truth.py"
python "$SCRIPT_DIR/scripts/server_preflight.py" --model "$MODEL_ARG"
python "$RUN_PY" paper-data
python "$RUN_PY" smoke-test
python "$RUN_PY" paper-train --variant CDTIR --task ner --model "$MODEL_ARG" --max-steps 5
python "$RUN_PY" paper-train --variant CDTIR --task relation --model "$MODEL_ARG" --max-steps 5
