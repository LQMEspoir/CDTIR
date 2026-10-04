#!/usr/bin/env bash
# ablation study (paper Table 10): CDTIR + 4 variants (CDTIR_NMoRE / CDTIR_F / CDTIR_EH / CDTIR_TO).
#   all on qwen2.5-7b (local HF cache); data uses data_end/data_merged (hardcoded in paper.py).
#   CDTIR (full) reuses the backbone model's trained NER router + RE adapter (no retraining), so
#   Table 10's CDTIR / EH / TO NER numbers match Table 9 exactly. Other variants reuse CDTIR's
#   NER/RE; CDTIR_F is zero-shot (no training).
#   artifacts under outputs/paper/<variant>/; final table at outputs/paper/reports/table_10.csv.
set -e
cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com

MODEL="${MODEL:-qwen2.5-7b}"
EMPTY_REL_WEIGHT="${EMPTY_REL_WEIGHT:-0.1}"
PAPER=outputs/paper

echo "==================== 1. prepare each variant (data_end/data_merged) ===================="

echo "===== 1.1 CDTIR (full = backbone): reuse backbone NER router + RE adapter ====="
# The full CDTIR model is the backbone model itself. Reuse its trained artifacts instead of
# retraining, so Table 10's CDTIR / EH / TO NER matches the backbone comparison (Table 9).
BB="outputs/backbone_$MODEL"
if [ ! -d "$BB/ner" ] || [ ! -d "$BB/re" ]; then
  echo "ERROR: backbone artifacts not found at $BB; run run_backbone.sh first." >&2
  exit 1
fi
mkdir -p "$PAPER/CDTIR"
rm -rf "$PAPER/CDTIR/ner" "$PAPER/CDTIR/relation"
ABS="$(pwd)"
ln -s "$ABS/$BB/ner" "$PAPER/CDTIR/ner"
ln -s "$ABS/$BB/re" "$PAPER/CDTIR/relation"

echo "===== 1.2 CDTIR_NMoRE (NER uses plain LoRA, RE reuses CDTIR) ====="
python run.py paper-train --variant CDTIR_NMoRE --task ner --model "$MODEL" --empty-relation-weight "$EMPTY_REL_WEIGHT"

echo "===== 1.3 CDTIR_EH (RE uses entity-aware hard prompt, NER reuses CDTIR) ====="
python run.py paper-train --variant CDTIR_EH --task relation --model "$MODEL" --empty-relation-weight "$EMPTY_REL_WEIGHT"

echo "===== 1.4 CDTIR_TO (RE uses text-only prompt, NER reuses CDTIR) ====="
python run.py paper-train --variant CDTIR_TO --task relation --model "$MODEL" --empty-relation-weight "$EMPTY_REL_WEIGHT"

echo "===== 1.5 CDTIR_F (zero-shot, no training) skip ====="

echo ""
echo "==================== 2. each variant NER individual prediction (-> predictions/ner.jsonl) ===================="
for v in CDTIR:router CDTIR_NMoRE:lora CDTIR_F:zero CDTIR_EH:reuse_cdtir CDTIR_TO:reuse_cdtir; do
  variant="${v%%:*}"; kind="${v##*:}"
  case "$kind" in
    router)
      python run.py predict --task ner --model "$MODEL" \
        --router-dir "$PAPER/CDTIR/ner" --router-generation-mode weighted \
        --dataset-file data_end/data_merged/benchmark/ner_test.jsonl --prompt-profile paper_ner \
        --output "$PAPER/$variant/predictions/ner.jsonl" ;;
    lora)
      python run.py predict --task ner --model "$MODEL" \
        --adapter "$PAPER/CDTIR_NMoRE/ner/final_adapter" \
        --dataset-file data_end/data_merged/benchmark/ner_test.jsonl --prompt-profile paper_ner \
        --output "$PAPER/$variant/predictions/ner.jsonl" ;;
    zero)
      python run.py predict --task ner --model "$MODEL" \
        --dataset-file data_end/data_merged/benchmark/ner_test.jsonl --prompt-profile paper_ner \
        --output "$PAPER/$variant/predictions/ner.jsonl" ;;
    reuse_cdtir)
      mkdir -p "$PAPER/$variant/predictions"
      cp "$PAPER/CDTIR/predictions/ner.jsonl" "$PAPER/$variant/predictions/ner.jsonl" ;;
  esac
  echo "  ✓ $variant NER prediction done"
done

echo ""
echo "==================== 3. CDTIR / CDTIR_NMoRE / CDTIR_F NER->RE pipeline ===================="

echo "===== 3.1 CDTIR ====="
python run.py cdtir-predict --variant CDTIR --model "$MODEL" \
  --ner-router "$PAPER/CDTIR/ner" \
  --re-adapter "$PAPER/CDTIR/relation/final_adapter" \
  --re-prompt-profile paper_re_e

echo "===== 3.2 CDTIR_NMoRE (RE reuses CDTIR) ====="
python run.py cdtir-predict --variant CDTIR_NMoRE --model "$MODEL" \
  --ner-adapter "$PAPER/CDTIR_NMoRE/ner/final_adapter" \
  --re-adapter "$PAPER/CDTIR/relation/final_adapter" \
  --re-prompt-profile paper_re_e

echo "===== 3.3 CDTIR_F (zero-shot) ====="
python run.py cdtir-predict --variant CDTIR_F --model "$MODEL" \
  --re-prompt-profile paper_re_e

echo ""
echo "==================== 4. CDTIR_EH / CDTIR_TO RE prediction (-> predictions/relation.jsonl) ===================="
for v in CDTIR_EH:paper_re_eh CDTIR_TO:paper_re_to; do
  variant="${v%%:*}"; prompt="${v##*:}"
  python run.py predict --task relation --model "$MODEL" \
    --adapter "$PAPER/$variant/relation/final_adapter" \
    --prompt-profile "$prompt" \
    --dataset-file "$PAPER/CDTIR/pipeline/re_e_gold_input.jsonl" \
    --output "$PAPER/$variant/predictions/relation.jsonl"
  echo "  ✓ $variant RE prediction done"
done

echo ""
echo "==================== 5. generate ablation table (table_10.csv) ===================="
python run.py paper-report

echo "==================== done: ablation table at $PAPER/reports/table_10.csv ===================="
