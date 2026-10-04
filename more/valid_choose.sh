#!/usr/bin/env bash
# run final prediction + evaluation on the validation split for 6 backbones, to select the backbone by validation.
# each backbone runs in order: NER (Router-MultiLoRA) -> RE-Pred (real pipeline) -> RE-Gold (upper bound),
# evaluate right after each prediction. After all, aggregate 12 micro metrics into a ranking and output valid_ranking.csv.
#
# output dir: outputs/valid_choose/<model>/  (prediction jsonl + *_eval metrics)
# summary file: outputs/valid_choose/valid_ranking.csv
# log file: outputs/valid_choose/run.log

set -u

MAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MAIN_DIR"

OUT="$MAIN_DIR/outputs/valid_choose"
VALID_NER="$MAIN_DIR/data_end/data_merged/benchmark/ner_valid.jsonl"
VALID_RE="$MAIN_DIR/data_end/data_merged/benchmark/relation_valid_e.jsonl"

# 6 backbones consistent with the test results (registry keys; order = ranking input order)
MODELS=(qwen2.5-7b qwen3-8b glm4-9b llama3.1-8b mistral7b-v0.3 baichuan2-7b)

# huggingface.co is unreachable; use the hf-mirror.com mirror to download base weights
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# data disk has ~40G free; 6 bases total ~92G and can't coexist: delete each model's HF cache after it finishes
declare -A MODEL_REPO=(
  [qwen2.5-7b]="Qwen/Qwen2.5-7B-Instruct"
  [qwen3-8b]="Qwen/Qwen3-8B"
  [glm4-9b]="zai-org/glm-4-9b-chat-hf"
  [llama3.1-8b]="meta-llama/Meta-Llama-3.1-8B-Instruct"
  [mistral7b-v0.3]="mistralai/Mistral-7B-Instruct-v0.3"
  [baichuan2-7b]="baichuan-inc/Baichuan2-7B-Chat"
)

# real HF cache dir (~/.cache/huggingface/hub is a symlink to it)
HF_HUB="$HOME/.cache/huggingface/hub"

cache_dir() { echo "$HF_HUB/models--${1//\//--}"; }

# explicit download + 8 retries + switch offline after (same pattern as run_lora.sh)
download_model() {
  local repo="${MODEL_REPO[$1]:-}"
  [ -n "$repo" ] || { echo "[FAIL] unknown model $1，no repo mapping"; return 1; }
  unset HF_HUB_OFFLINE   # reset, to avoid the previous round's offline mode blocking this round's download
  echo "========== $1 download base $repo =========="
  local ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf download "$repo"; then ok=1; break; fi
    echo "download failed (attempt $attempt), retrying in 30s..."; sleep 30
  done
  [ "$ok" = "1" ] || { echo "[FAIL] $repo failed 8 times"; return 1; }
  export HF_HUB_OFFLINE=1
  return 0
}

cleanup_model() {
  local repo="${MODEL_REPO[$1]:-}"
  [ -n "$repo" ] || return 0
  local d; d="$(cache_dir "$repo")"
  if [ -e "$d" ]; then
    rm -rf "$d" && echo "deleted base cache: $d"
  fi
}

mkdir -p "$OUT"
MASTER_LOG="$OUT/run.log"

# ---------- preflight: validation set + each model's artifacts ----------
echo "=== preflight ==="
[ -f "$VALID_NER" ] || { echo "MISSING: $VALID_NER"; exit 1; }
[ -f "$VALID_RE" ] || { echo "MISSING: $VALID_RE"; exit 1; }
for m in "${MODELS[@]}"; do
  [ -f "outputs/backbone_$m/ner/router_manifest.json" ] || echo "WARN: $m missing NER router package"
  [ -f "outputs/backbone_$m/re/final_adapter/adapter_config.json" ] || echo "WARN: $m missing RE adapter"
done
echo "preflight OK, valid_ner=$(wc -l < "$VALID_NER") valid_re=$(wc -l < "$VALID_RE")"

# ---------- per-model full pipeline ----------
run_one() {
  local m="$1" d="$OUT/$m"
  mkdir -p "$d/ner_eval" "$d/re_eval" "$d/re_gold_eval"

  download_model "$m" || return 1

  echo "========== $m : 1/6 NER predict =========="
  python run.py predict --task ner --model "$m" \
    --router-dir "outputs/backbone_$m/ner" --router-generation-mode weighted \
    --prompt-profile paper_ner \
    --dataset-file "$VALID_NER" \
    --output "$d/ner_predictions.jsonl" || { echo "[FAIL] $m ner predict"; return 1; }

  echo "========== $m : 2/6 NER evaluate =========="
  python run.py evaluate --task ner --predictions "$d/ner_predictions.jsonl" \
    --output-dir "$d/ner_eval" || { echo "[FAIL] $m ner eval"; return 1; }

  echo "========== $m : 3/6 build RE-Pred input =========="
  python - "$d" "$VALID_RE" <<'PY'
import json, sys
d, rel = sys.argv[1], sys.argv[2]
ner = [json.loads(l) for l in open(f"{d}/ner_predictions.jsonl", encoding="utf-8") if l.strip()]
rel_gold = {r["text_id"]: r for r in (json.loads(l) for l in open(rel, encoding="utf-8") if l.strip())}
rows = [{"text_id": r["text_id"], "input": rel_gold[r["text_id"]]["input"],
         "entities": r.get("prediction", []), "output": rel_gold[r["text_id"]]["output"]} for r in ner]
with open(f"{d}/re_e_pred_input.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"built {len(rows)} pred-input rows")
PY

  echo "========== $m : 4/6 RE-Pred predict =========="
  python run.py predict --task relation --model "$m" \
    --adapter "outputs/backbone_$m/re/final_adapter" \
    --prompt-profile paper_re_e \
    --dataset-file "$d/re_e_pred_input.jsonl" \
    --output "$d/re_e_pred_predictions.jsonl" || { echo "[FAIL] $m re-pred predict"; return 1; }

  echo "========== $m : 5/6 RE-Pred evaluate =========="
  python run.py evaluate --task relation --predictions "$d/re_e_pred_predictions.jsonl" \
    --output-dir "$d/re_eval" || { echo "[FAIL] $m re-pred eval"; return 1; }

  echo "========== $m : 6/6 RE-Gold predict + evaluate =========="
  python run.py predict --task relation --model "$m" \
    --adapter "outputs/backbone_$m/re/final_adapter" \
    --prompt-profile paper_re_e \
    --dataset-file "$VALID_RE" \
    --output "$d/re_e_gold_predictions.jsonl" || { echo "[FAIL] $m re-gold predict"; return 1; }

  python run.py evaluate --task relation --predictions "$d/re_e_gold_predictions.jsonl" \
    --output-dir "$d/re_gold_eval" || { echo "[FAIL] $m re-gold eval"; return 1; }

  return 0
}

# ---------- main loop: tee all output to the tmux screen and run.log ----------
{
  echo "start: $(date)"
  for m in "${MODELS[@]}"; do
    echo "################ $m ################"
    run_one "$m" || echo "[SKIP] $m failed, continue to next"
    cleanup_model "$m"
  done
  echo "end: $(date)"

  echo "=== aggregate ranking ==="
  python - "$OUT" "${MODELS[@]}" <<'PY'
import json, sys
out = sys.argv[1]; models = sys.argv[2:]

M = {}  # metric name -> {model: value}
def add(name, m, v): M.setdefault(name, {})[m] = v

for m in models:
    ner = json.load(open(f"{out}/{m}/ner_eval/ner_predictions_metrics.json"))
    rp  = json.load(open(f"{out}/{m}/re_eval/re_e_pred_predictions_metrics.json"))
    rg  = json.load(open(f"{out}/{m}/re_gold_eval/re_e_gold_predictions_metrics.json"))
    add("NER-P", m, ner["micro"]["precision"]); add("NER-R", m, ner["micro"]["recall"])
    add("NER-F1", m, ner["micro"]["f1"]);      add("NER-Acc", m, ner["accuracy"])
    add("RE-P", m, rp["micro"]["precision"]); add("RE-R", m, rp["micro"]["recall"])
    add("RE-F1", m, rp["micro"]["f1"]);      add("RE-Acc", m, rp["accuracy"])
    add("RE-Gold-P", m, rg["micro"]["precision"]); add("RE-Gold-R", m, rg["micro"]["recall"])
    add("RE-Gold-F1", m, rg["micro"]["f1"]);      add("RE-Gold-Acc", m, rg["accuracy"])

names = list(M.keys())
ranks = {m: [] for m in models}
norms = {m: [] for m in models}
wins = {m: 0 for m in models}
for name in names:
    vals = M[name]; vmin, vmax = min(vals.values()), max(vals.values())
    ordered = sorted(models, key=lambda m: -vals[m])
    prev_v, prev_r = None, 0
    for i, m in enumerate(ordered):
        if vals[m] != prev_v:
            prev_r = i + 1
        ranks[m].append(prev_r); prev_v = vals[m]
        norms[m].append((vals[m]-vmin)/(vmax-vmin) if vmax > vmin else 1.0)
        if prev_r == 1: wins[m] += 1

rows = []
for m in models:
    ar = sum(ranks[m])/len(ranks[m]); an = sum(norms[m])/len(norms[m])
    rows.append((ar, -an, m))
rows.sort()

csv_path = f"{out}/valid_ranking.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    header = "rank,model,avg_rank,avg_norm,#1," + ",".join(names)
    f.write(header + "\n")
    for i, (ar, _nan, m) in enumerate(rows, 1):
        an = -_nan
        f.write(f"{i},{m},{ar:.3f},{an:.3f},{wins[m]}," + ",".join(f"{M[n][m]:.3f}" for n in names) + "\n")

print("\n===== validation overall ranking =====")
print(f"{'#':<3}{'model':<16}{'avg_rank':>10}{'avg_norm':>10}{'#1':>4}")
for i, (ar, _nan, m) in enumerate(rows, 1):
    print(f"{i:<3}{m:<16}{ar:>10.3f}{-_nan:>10.3f}{wins[m]:>4}")
print(f"\nCSV written: {csv_path}")
print("best model per metric:")
for name in names:
    best = max(models, key=lambda m: M[name][m])
    print(f"  {name:<10} -> {best}  {M[name][best]:.3f}")
PY
} 2>&1 | tee "$MASTER_LOG"

echo "all done. summary at $OUT/valid_ranking.csv, log at $MASTER_LOG"
