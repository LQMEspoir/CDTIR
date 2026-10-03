# -*- coding: utf-8 -*-
"""Targeted supplementation of "Product" entities (without rerunning NER).

Background: NER's "Product" recall is only ~17%, the root cause dragging down RE (Experience / Budget Limit).
This script runs Ollama recognition only for "Product" and appends the newly found product entities into
data_end/data_expanded/ner/entity_all.jsonl — only appending "Product", never overwriting or deleting existing entities.

Resumable: progress is written to product_supplement_progress.jsonl; reruns skip already-supplemented text_ids.
Usage:python supplement_products.py [--model qwen2.5:7b] [--limit N]
"""
import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NER_FILE = ROOT / "data_end/data_expanded" / "ner" / "entity_all.jsonl"
PROGRESS = ROOT / "data_end/data_expanded" / "ner" / "product_supplement_progress.jsonl"

# Chinese prompt. English translation: "You are an expert in recognizing 'Product' entities in travel text; recognize all 'Product'-type entities from the given text."
PRODUCT_PROMPT = """你是一个旅游文本「产品」实体识别专家。请从给定文本中识别所有「产品」类实体。

「产品」= 旅行过程中想要体验的活动、项目或服务。例如：骑马、看日出、拍照、漂流、温泉、滑雪、钓鱼、划船、徒步、露营、采摘、看演出、热气球、祈福、观鸟。

严格排除（这些不是产品，不要标）：
- 餐饮：火锅、小吃、美食、牛肉面、湘菜
- 住宿：民宿、酒店、客栈、海景房
- 交通：自驾、索道、游船、巴士、飞机、轮渡、步行
- 目的地：景点名、地名（黄山、西湖、丽江古城）
- 时间/时长：几天、下午、周末
- 其他：预算、人数、人群、强度、人气、天气

请只输出 JSON，格式为 {"entities": [{"entity_text": "...", "entity_label": "产品"}]}。找不到产品时输出 {"entities": []}。"""

PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_text": {"type": "string"},
                    "entity_label": {"type": "string", "enum": ["产品"]},
                },
                "required": ["entity_text", "entity_label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def ollama_products(url: str, model: str, text: str, timeout: int) -> list[dict]:
    payload = {
        "model": model,
        "stream": False,
        "format": PRODUCT_SCHEMA,
        "messages": [
            {"role": "system", "content": PRODUCT_PROMPT},
            {"role": "user", "content": "文本：\n" + text},
        ],
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},
        "keep_alive": "30m",
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    raw = body.get("message", {}).get("content", "")
    parsed = json.loads(raw)
    out, seen = [], set()
    for e in parsed.get("entities", []):
        t = str(e.get("entity_text", "")).strip()
        if t and t not in seen:
            seen.add(t)
            out.append({"entity_text": t, "entity_label": "产品"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="process at most N rows; 0 = all")
    args = ap.parse_args()

    rows = read_jsonl(NER_FILE)
    done = {r["text_id"] for r in read_jsonl(PROGRESS)} if PROGRESS.exists() else set()
    print(f"entity_all {len(rows)} rows, supplemented {len(done)} rows")

    t0 = time.time()
    added_total = 0
    for i, r in enumerate(rows, 1):
        tid = r.get("text_id")
        if tid in done:
            continue
        if args.limit and i > args.limit:
            break
        text = r["input"]
        if text.startswith("文本:"):
            text = text[len("文本:"):]

        products = []
        for attempt in range(1, args.retries + 1):
            try:
                products = ollama_products(args.url, args.model, text, args.timeout)
                break
            except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
                if attempt < args.retries:
                    time.sleep(min(2 ** attempt, 8))
                else:
                    print(f"[{i}/{len(rows)}] {tid} product recognition failed: {exc}", flush=True)

        # only add product-class samples, dedup
        existing = {(e["entity_text"], e["entity_label"]) for e in r.get("output", [])}
        added = 0
        for p in products:
            key = (p["entity_text"], "产品")
            if key not in existing:
                r["output"].append(p)
                existing.add(key)
                added += 1
        added_total += added
        append_jsonl(PROGRESS, {"text_id": tid, "added_products": added})

        if i % 10 == 0:
            print(f"[{i}/{len(rows)}] took {time.time()-t0:.0f}s, cumulative new products {added_total}", flush=True)

    # write back (back up first)
    shutil.copy(NER_FILE, NER_FILE.with_suffix(".jsonl.bak_before_product"))
    write_jsonl(NER_FILE, rows)
    print(f"\ndone: cumulative new product entities {added_total}  -> {NER_FILE}")
    print(f"backup -> {NER_FILE.name}.bak_before_product")


if __name__ == "__main__":
    main()
