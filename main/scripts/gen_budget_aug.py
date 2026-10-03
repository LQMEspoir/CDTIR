#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate budget-entity samples with a local qwen2.5-7b for NER data augmentation.

Cover various budget expressions: exact amounts, ranges, per-capita, budget-travel, save-money, high value-for-money.
Output to data_end/data_aug/budget_samples.jsonl, format aligned with entity_all.jsonl:
  {"text_id": ..., "input": "文本:...", "output": [{"entity_text": ..., "entity_label": "预算"}]}

Usage: python scripts/gen_budget_aug.py [num_rows] [--out output_path]
"""
import json
import os
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("CDTIR_QWEN25_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

# Chinese few-shot prompt. English translation:
#   "You are a travel data annotation assistant. Generate travel sentences and annotate 'Budget' entities in each.
#    Requirements: 1. Diverse wording, never reuse the same template. 2. Cover various budget expressions (exact/range/approximate/per-capita/budget-travel/money-saving/value-for-money/limited funds). 3. Vary destination, scenario, and style (question/statement/guide).
#    Output a JSON array: {"input": "text:...", "entities": [{"entity_text": "...", "entity_label": "Budget"}]}, with examples below. Generate 40 rows, output JSON directly."
FEWSHOT = """你是旅游数据标注助手。生成关于旅游的句子，每 rows标注「预算」实体。

要求：
1. 句子内容和句式必须多样，严禁用相同模板重复生成（不要总是"XX推荐，预算XX元"）。
2. 覆盖多种预算表达：具体金额（3000元、两千块）、区间（10000-15000元）、约数（5000左右、预算3000以内）、人均（人均2000元）、穷游、省钱、性价比高、资金有限、预算不多。
3. 目的地、场景、句式都要变化（提问式、陈述式、攻略式混合）。

输出 JSON 数组，每个元素格式：
{"input": "文本:...", "entities": [{"entity_text": "...", "entity_label": "预算"}]}

示例：
{"input": "文本:去云南玩，预算5000元", "entities": [{"entity_text": "5000元", "entity_label": "预算"}]}
{"input": "文本:学生党穷游成都重庆", "entities": [{"entity_text": "穷游", "entity_label": "预算"}]}
{"input": "文本:人均2000元的国内旅行推荐", "entities": [{"entity_text": "人均2000元", "entity_label": "预算"}]}
{"input": "文本:省钱攻略：3000块玩转三亚", "entities": [{"entity_text": "省钱", "entity_label": "预算"}, {"entity_text": "3000块", "entity_label": "预算"}]}
{"input": "文本:春节想出去玩，预算在一万五左右，去哪合适？", "entities": [{"entity_text": "一万五左右", "entity_label": "预算"}]}

请生成 40  rows。直接输出 JSON 数组，不要解释。"""

# validation: budget entity text must contain one of these expressions (amount with digits, or budget/budget-travel/save-money/value words)
BUDGET_PAT = re.compile(r'\d|预算|穷游|省钱|性价比|人均|资金|花费|开销')


def extract_json_array(text: str):
    """Extract the JSON array from model output."""
    text = text.strip()
    # strip markdown fences
    m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.S | re.I)
    if m:
        text = m.group(1).strip()
    # find the outermost [ ... ]
    l, r = text.find('['), text.rfind(']')
    if 0 <= l < r:
        try:
            return json.loads(text[l:r + 1])
        except Exception:
            pass
    return None


def validate(sample: dict) -> bool:
    ents = sample.get('entities', [])
    if not ents:
        return False
    for e in ents:
        if e.get('entity_label') != '预算':
            return False
        if not BUDGET_PAT.search(e.get('entity_text', '')):
            return False
    return True


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    out = ROOT / "data_end/data_aug" / "budget_samples.jsonl"

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()

    seen = set()
    kept = []
    for rnd in range(8):
        if len(kept) >= target:
            break
        temp = 0.75 + 0.1 * (rnd % 3)
        msgs = [{"role": "user", "content": FEWSHOT}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True,
                                     temperature=temp, top_p=0.9)
        resp = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        arr = extract_json_array(resp)
        if arr is None:
            print(f"round {rnd+1}  parse failed")
            continue
        for s in arr:
            if isinstance(s, dict) and validate(s) and s['input'] not in seen:
                seen.add(s['input'])
                kept.append(s)
        print(f"round {rnd+1}  (temp={temp}) accumulated {len(kept)}/{target}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for i, s in enumerate(kept[:target]):
            text = s['input']
            if not text.startswith('文本:'):
                text = '文本:' + text
            rec = {"text_id": f"aug_budget_{i:04d}", "input": text, "output": s['entities']}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"✅ final {len(kept[:target])}  rows → {out}")


if __name__ == "__main__":
    main()
