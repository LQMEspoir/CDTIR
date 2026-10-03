# File: src/tourism_ie/parsing.py
# Module responsibility: structured parsing of model-generated text: extracts a JSON payload from noisy output and normalizes it into a unified item list.
# Main data flow: raw generated string -> locate/tolerant-parse JSON fragment -> normalize entity or relation items.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
import json
import re
from typing import Any


def extract_json_payload(text: str) -> Any:
    """- Responsibility: locates and parses the JSON body from generated text that may include prose, Markdown fences or extra prefixes/suffixes.
    - Parameters: text: str.
    - Returns: return type is `Any`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s or s in {"没有找到任何实体", "找不到关系"}:
        # if not s or s in {"no entity found", "no relation found"}:
        return []
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.S | re.I)
    if fence:
        s = fence.group(1).strip()
    # Prefer full JSON array/object.
    for candidate in (s,):
        try:
            value = json.loads(candidate)
            return value if isinstance(value, list) else [value]
        except Exception:
            pass
    # Find the outermost array if model added prose.
    l, r = s.find("["), s.rfind("]")
    if 0 <= l < r:
        try:
            value = json.loads(s[l:r+1])
            return value if isinstance(value, list) else [value]
        except Exception:
            pass
    # Compatibility with original NER target: {...}{...}
    out, decoder, i = [], json.JSONDecoder(), 0
    while i < len(s):
        while i < len(s) and s[i].isspace(): i += 1
        if i >= len(s): break
        try:
            obj, j = decoder.raw_decode(s, i)
            out.append(obj); i = j
        except Exception:
            i += 1
    return out


def normalize_money(text: str) -> str:
    """- Responsibility: normalizes parsed objects into the task item list, dropping invalid or missing-field outputs.
    - Parameters: task: str,value: Any.
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Normalize budget amounts to pure numbers (strip units/modifiers, expand k/w/千/万 abbreviations)."""
    # Chinese budget/capita amount tokens: 预算/人均 = budget / per-capita, 控制在/不超过/大约/在 = controlled at / not over / about / at,
    # 千/万 = thousand / ten-thousand, 元/块 = yuan, 以内/左右/上下/以下/以上 = within / about / around / below / above.
    t = str(text).strip()
    t = re.sub(r'^(预算控制在|预算不超过|预算需要准备|预算大约|预算只有|预算在|人均花费大概|人均花费|人均消费|人均预算|预算|人均|控制在|不超过|大约|在|只有)', '', t)
    t = re.sub(r'(\d+)-(\d+(?:\.\d+)?)([kKwW千万元块rR])', r'\1\3-\2\3', t)
    t = re.sub(r'(\d+(?:\.\d+)?)[kK]', lambda m: str(int(float(m.group(1))*1000)), t)
    t = re.sub(r'(\d+(?:\.\d+)?)[wW]', lambda m: str(int(float(m.group(1))*10000)), t)
    t = re.sub(r'(\d+(?:\.\d+)?)千', lambda m: str(int(float(m.group(1))*1000)), t)
    t = re.sub(r'(\d+(?:\.\d+)?)万', lambda m: str(int(float(m.group(1))*10000)), t)
    t = re.sub(r'[元块rR]', '', t)
    t = re.sub(r'(以内|左右|上下|以下|以上|之间|内|元左右|块左右)$', '', t)
    return t.strip()


def normalize_items(task: str, value: Any) -> list[dict]:
    items = value if isinstance(value, list) else extract_json_payload(str(value))
    out=[]
    if task == "ner":
        for x in items:
            if isinstance(x, dict) and x.get("entity_text") and x.get("entity_label"):
                txt = str(x["entity_text"]).strip()
                label = str(x["entity_label"]).strip()
                if label == "预算":
                # if label == "Budget":
                    txt = normalize_money(txt)
                elif label == "人群":
                # elif label == "Crowd":
                    txt = re.sub(r'(的人群|的人)$', '', txt)
                elif label == "强度":
                # elif label == "Intensity":
                    txt = re.sub(r'(徒步|散步|步行|步|游)$', '', txt)
                out.append({"entity_text": txt, "entity_label": label})
    else:
        for x in items:
            if not isinstance(x, dict): continue
            s=x.get("subject", x.get("ob1")); p=x.get("predicate", x.get("rel")); o=x.get("object", x.get("ob2"))
            if s and p and o:
                s=str(s).strip(); p=str(p).strip(); o=str(o).strip()
                if p == "预算限制":
                # if p == "Budget Limit":
                    o = normalize_money(o)
                elif p == "人均预算":
                # elif p == "Per-capita Budget":
                    s = normalize_money(s)
                out.append({"subject":s,"predicate":p,"object":o})
    return out
