#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com

# clean up last run's leftovers (mistral left when zero-shot was killed)
rm -rf $HOME/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3

# one by one: download -> zero-shot NER/RE -> evaluate -> delete weights (keep outputs)
# config: NER=paper_ner (default), RE=paper_re_e (no few-shot); Qwen3 thinking mode is already disabled in code
for model in qwen3-8b baichuan2-7b qwen2.5-7b glm4-9b llama3.1-8b mistral7b-v0.3; do
  convert=0
  case "$model" in
    qwen3-8b)        repo="Qwen/Qwen3-8B";                         cache="models--Qwen--Qwen3-8B" ;;
    baichuan2-7b)    repo="baichuan-inc/Baichuan2-7B-Chat";        cache="models--baichuan-inc--Baichuan2-7B-Chat"; convert=1 ;;
    qwen2.5-7b)      repo="Qwen/Qwen2.5-7B-Instruct";              cache="models--Qwen--Qwen2.5-7B-Instruct" ;;
    glm4-9b)         repo="zai-org/glm-4-9b-chat-hf";              cache="models--zai-org--glm-4-9b-chat-hf" ;;
    llama3.1-8b)     repo="meta-llama/Meta-Llama-3.1-8B-Instruct"; cache="models--meta-llama--Meta-Llama-3.1-8B-Instruct" ;;
    mistral7b-v0.3)  repo="mistralai/Mistral-7B-Instruct-v0.3";    cache="models--mistralai--Mistral-7B-Instruct-v0.3" ;;
    deepseek-7b)     repo="deepseek-ai/deepseek-llm-7b-chat";      cache="models--deepseek-ai--deepseek-llm-7b-chat" ;;
  esac

  echo "==================== $model ===================="
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf download "$repo"; then ok=1; break; fi
    echo "download failed (attempt $attempt), retrying in 30s..."
    sleep 30
  done
  if [ "$ok" != "1" ]; then echo "download $repo failed 8 times, exiting"; exit 1; fi
  if [ "$convert" = "1" ]; then
    echo "Baichuan2 converting to safetensors (bypass torch<2.6 pickle limit)..."
    python convert_baichuan_safetensors.py "$HOME/.cache/huggingface/hub/$cache"
  fi

  out="outputs/zero-shot/$model"
  HF_HUB_OFFLINE=1 python run.py cdtir-predict --variant CDTIR_F \
    --model "$model" --re-prompt-profile paper_re_e \
    --output-dir "$out/pipeline" --max-new-tokens 256
  HF_HUB_OFFLINE=1 python run.py evaluate --task ner \
    --predictions "$out/pipeline/ner_predictions.jsonl" --output-dir "$out/ner_eval"
  HF_HUB_OFFLINE=1 python run.py evaluate --task relation \
    --predictions "$out/pipeline/re_e_pred_predictions.jsonl" --output-dir "$out/re_eval"
  HF_HUB_OFFLINE=1 python run.py evaluate --task relation \
    --predictions "$out/pipeline/re_e_gold_predictions.jsonl" --output-dir "$out/re_gold_eval"

  echo "===== delete weights $model ====="
  rm -rf "$HOME/.cache/huggingface/hub/$cache"
done

