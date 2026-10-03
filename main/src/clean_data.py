# -*- coding: utf-8 -*-
"""Clean raw crawled data in data_end/data_more: unify format, dedup, filter question-like rows, normalize.

Output: cleaned_intent.jsonl (unlabeled skeleton {input, output:[]}) for later annotation.
"""
import json
import re
import csv
from pathlib import Path
import openpyxl

BASE = Path(__file__).resolve().parent.parent / "data_end" / "data_more"

# intent / question keywords (Chinese: ask / recommend / how / how-to / plan / guide / arrange / route /
# itinerary / where / please-ask / help / suitable / fit / which / what / can / how-to-go / how-to-play)
INTENT_WORDS = ["求", "推荐", "怎么", "如何", "规划", "攻略", "安排", "路线",
                "行程", "哪里", "请问", "帮忙", "合适", "适合", "哪些", "什么",
                "能否", "可以", "怎么走", "怎么去", "怎么玩"]


def normalize(text: str) -> str:
    """Strip newlines, remove emoji/special symbols, collapse whitespace."""
    if not text:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    # keep Chinese, English, digits and common CJK/ASCII punctuation
    text = re.sub(
        r"[^一-鿿　-〿＀-￯a-zA-Z0-9，。！？、；：（）【】《》“”‘’%#@\-—~. ]",
        "", text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_intent(text: str) -> bool:
    if not text:
        return False
    if "？" in text or "?" in text:
        return True
    return any(w in text for w in INTENT_WORDS)


def read_xiaohongshu_txt() -> list[str]:
    """Xiaohongshu Q&A .txt: the first line is the "text" header; each entry may span multiple lines, separated by blank lines."""
    rows, cur = [], []
    for line in (BASE / "xiaohongshu_qa.txt").open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                rows.append(" ".join(cur))
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        rows.append(" ".join(cur))
    if rows and rows[0].strip() == "文本":
        rows = rows[1:]
    return rows


def main():
    all_rows = []  # {source, text}

    # 1) Ctrip csv (field 1)
    with (BASE / "ctrip_travel_qa.csv").open(encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0].strip():
                all_rows.append({"source": "Ctrip", "text": row[0].strip()})

    # 2) Mafengwo txt (field 1_文本 = the "text" column)
    lines = [l.strip() for l in (BASE / "mafengwo_qa.txt").open(encoding="utf-8") if l.strip()]
    if lines and ("文本" in lines[0]):
        lines = lines[1:]
    for l in lines:
        all_rows.append({"source": "Mafengwo", "text": l})

    # 3) Xiaohongshu txt - removed, unused
    # for t in read_xiaohongshu_txt():
    #     all_rows.append({"source": "Xiaohongshu_txt", "text": t})   # "小红书txt" = Xiaohongshu source (removed, unused)

    # 4) Xiaohongshu-2 xlsx (title + body)
    wb = openpyxl.load_workbook(BASE / "xiaohongshu-2.xlsx", read_only=True)
    ws = wb.active
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r and r[1]:
            text = f"{r[0] or ''} {r[1]}".strip()
            all_rows.append({"source": "Xiaohongshu_xlsx", "text": text})

    for s in ["Ctrip", "Mafengwo", "Xiaohongshu_txt", "Xiaohongshu_xlsx"]:
        n = sum(1 for x in all_rows if x["source"] == s)
        print(f"  {s:12s}: {n}")

    # normalize + dedup
    seen, deduped = set(), []
    for x in all_rows:
        t = normalize(x["text"])
        if not t or len(t) < 5:
            continue
        if t in seen:
            continue
        seen.add(t)
        x["text"] = t
        deduped.append(x)
    print(f"normalize+dedup: {len(deduped)} rows")

    # filter question-like rows
    intent = [x for x in deduped if is_intent(x["text"])]
    print(f"question/intent: {len(intent)} rows")
    for s in ["Ctrip", "Mafengwo", "Xiaohongshu_txt", "Xiaohongshu_xlsx"]:
        n = sum(1 for x in intent if x["source"] == s)
        print(f"  {s:12s}: {n}")

    # output unlabeled skeleton JSONL
    out = BASE / "cleaned_intent.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for x in intent:
            f.write(json.dumps({"input": f"文本:{x['text']}", "output": []}, ensure_ascii=False) + "\n")
    print(f"written: {out} ({len(intent)} unlabeled skeleton rows)")


if __name__ == "__main__":
    main()
