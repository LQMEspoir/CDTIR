"""Build full relation truth from the unique NER corpus with Ollama.

Existing reviewed relation rows are preserved. Rows that cannot form any supported
relation from their gold entity types become explicit negative examples. Only the
remaining unresolved rows are sent to Ollama. Progress is append-only and resumable.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("CDTIR_ROOT",Path.cwd())).expanduser().resolve()
RELATIONS = ["游玩顺序", "出行时间", "停留时长", "预算限制", "人均预算", "体验内容"]
# RELATIONS = ["Play Order", "Travel Time", "Stay Duration", "Budget Limit", "Per-capita Budget", "Experience"]
ALLOWED = {
    "体验内容": ({"目的地", "餐饮", "住宿"}, {"产品"}),
    # "体验内容": ({"Destination", "Dining", "Lodging"}, {"Product"})
    "预算限制": ({"目的地", "餐饮", "住宿", "产品", "交通"}, {"预算"}),
    # "预算限制": ({"Destination", "Dining", "Lodging", "Product", "Transport"}, {"Budget"})
    "人均预算": ({"预算"}, {"人数"}),
    # "人均预算": ({"Budget"}, {"People"})
    "停留时长": ({"目的地", "餐饮", "住宿", "产品"}, {"时长"}),
    # "停留时长": ({"Destination", "Dining", "Lodging", "Product"}, {"Duration"})
    "出行时间": ({"出发地", "返程地", "目的地", "餐饮", "住宿", "产品"}, {"时间"}),
    # "出行时间": ({"Origin", "Return", "Destination", "Dining", "Lodging", "Product"}, {"Time"})
}
ORDER_PAIRS = {("出发地", "目的地"), ("目的地", "目的地"), ("目的地", "返程地")}
# ORDER_PAIRS = {("Origin", "Destination"), ("Destination", "Destination"), ("Destination", "Return")}

SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string", "enum": RELATIONS},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relations"],
    "additionalProperties": False,
}

# Chinese relation-extraction draft prompt. English translation:
#   "You are a relation extraction expert. Extract all valid triples (head entity, relation, tail entity) from the text.
#    Relation types only: Play Order, Travel Time, Stay Duration, Budget Limit, Per-capita Budget, Experience.
#    Only extract relations explicitly expressed in the text (no common-sense guessing). Return an empty array if no relation. Examples (5 below)."
RELATION_DRAFT_PROMPT = """你是一位关系抽取专家。请从文本中提取所有有效三元组（头实体、关系、尾实体）。
关系类型仅限：游玩顺序、出行时间、停留时长、预算限制、人均预算、体验内容。
只提取文本明确表达的关系，不得依靠常识猜测。没有关系时返回空数组，不要输出解释。

示例1：
文本：从北京出发去上海玩3天
输出：{"relations":[{"subject":"北京","predicate":"游玩顺序","object":"上海"},{"subject":"上海","predicate":"停留时长","object":"3天"}]}

示例2：
文本：10月1日从成都自驾去九寨沟
输出：{"relations":[{"subject":"成都","predicate":"游玩顺序","object":"九寨沟"},{"subject":"九寨沟","predicate":"出行时间","object":"10月1日"}]}

示例3：
文本：300元左右的酒店推荐
输出：{"relations":[{"subject":"酒店","predicate":"预算限制","object":"300元"}]}

示例4：
文本：去草原想骑马，求推荐
输出：{"relations":[{"subject":"草原","predicate":"体验内容","object":"骑马"}]}
注意：「体验内容」的尾实体（object）必须是「产品」类型的活动/体验（如骑马、看日出、拍照、徒步、划船），不能是住宿（民宿/酒店）或餐饮（火锅/小吃）——住店、吃饭属于住宿/餐饮偏好，不构成体验内容关系。

示例5：
文本：三个人去三亚，人均预算3000元
输出：{"relations":[{"subject":"3000元","predicate":"人均预算","object":"3"}]}"""

# Chinese review prompt. English translation:
#   "You are an entity and relation review expert. Check and correct inconsistencies, fill missing triples, remove relations violating constraints.
#    Only keep relations explicitly expressed (no guessing); do not create/rewrite entities.
#    Relations only: Play Order, Travel Time, Stay Duration, Budget Limit, Per-capita Budget, Experience.
#    Entity-type combinations: Experience=(Destination/Dining/Lodging, Product); Budget Limit=(Destination/Dining/Lodging/Product/Transport, Budget); Per-capita Budget=(Budget, People); Stay Duration=(Destination/Dining/Lodging/Product, Duration); Travel Time=(Origin/Return/Destination/Dining/Lodging/Product, Time); Play Order=(Origin,Destination)/(Destination,Destination)/(Destination,Return).
#    subject/object must copy entity_text verbatim. Return empty array if no relation."
REVIEW_SYSTEM_PROMPT = """你是一位实体与关系校对专家。请基于给定文本、实体列表和关系抽取草稿，核查并修正不一致项，补全遗漏三元组，删除不符合约束的关系。
只能保留文本明确表达的关系，不得依靠常识猜测，不得创造或改写实体。
关系仅限：游玩顺序、出行时间、停留时长、预算限制、人均预算、体验内容。
实体类型组合：
- 体验内容：(目的地/餐饮/住宿, 产品)——object 只能是「产品」类型的活动/体验（骑马、看日出、拍照、徒步等），不能是住宿（民宿/酒店）或餐饮（火锅/小吃）
- 预算限制：(目的地/餐饮/住宿/产品/交通, 预算)
- 人均预算：(预算, 人数)
- 停留时长：(目的地/餐饮/住宿/产品, 时长)
- 出行时间：(出发地/返程地/目的地/餐饮/住宿/产品, 时间)
- 游玩顺序：(出发地,目的地)、(目的地,目的地)、(目的地,返程地)
补全遗漏三元组时，请特别留意「体验内容」「人均预算」「预算限制」等容易被遗漏的关系，凡是文本明确表达的都提取出来。
subject和object必须逐字复制实体列表中的entity_text。文本没有明确关系时返回空数组。不要输出解释。"""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def text_id(text: str) -> str:
    return "txt_" + hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def merge_ner(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        text = row["input"].strip()
        if text not in grouped:
            order.append(text)
        grouped[text].append(row)
    merged = []
    for text in order:
        seen = set()
        entities = []
        for row in grouped[text]:
            for entity in row.get("output", []):
                key = (entity.get("entity_text"), entity.get("entity_label"))
                if all(key) and key not in seen:
                    seen.add(key)
                    entities.append({"entity_text": key[0], "entity_label": key[1]})
        merged.append({"text_id": text_id(text), "input": text, "output": entities})
    return merged


def correction_map() -> dict[str, str]:
    path = ROOT / "data/cleaning/ner_corrections.json"
    result = {}
    if path.exists():
        for item in json.loads(path.read_text(encoding="utf-8")):
            if item.get("new_input"):
                result[item["match_input"]] = item["new_input"]
    return result


def existing_relation_map() -> dict[str, list[dict]]:
    corrections = correction_map()
    result = {}
    for row in read_jsonl(ROOT / "data_end/data_generate/relation/relation_all.jsonl"):
        text = corrections.get(row["input"], row["input"])
        result[text] = row.get("output", [])
    return result


def labels_by_text(entities: list[dict]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        result[entity["entity_text"]].add(entity["entity_label"])
    return result


def pair_allowed(predicate: str, subject_labels: set[str], object_labels: set[str]) -> bool:
    if predicate == "游玩顺序":
        return any((a, b) in ORDER_PAIRS for a in subject_labels for b in object_labels)
    if predicate not in ALLOWED:
        return False
    left, right = ALLOWED[predicate]
    return bool(subject_labels & left) and bool(object_labels & right)


def has_candidate_pair(entities: list[dict]) -> bool:
    by_text = labels_by_text(entities)
    names = list(by_text)
    for subject in names:
        for obj in names:
            for predicate in RELATIONS:
                if pair_allowed(predicate, by_text[subject], by_text[obj]):
                    return True
    return False


def validate_relations(relations: object, entities: list[dict]) -> tuple[list[dict], list[dict]]:
    by_text = labels_by_text(entities)
    valid, rejected, seen = [], [], set()
    if not isinstance(relations, list):
        return [], [{"reason": "relations_not_list", "value": relations}]
    for relation in relations:
        if not isinstance(relation, dict):
            rejected.append({"reason": "relation_not_object", "value": relation})
            continue
        triple = {
            "subject": str(relation.get("subject", "")).strip(),
            "predicate": str(relation.get("predicate", "")).strip(),
            "object": str(relation.get("object", "")).strip(),
        }
        reason = None
        if triple["subject"] not in by_text or triple["object"] not in by_text:
            reason = "endpoint_not_in_gold_entities"
        elif not pair_allowed(triple["predicate"], by_text[triple["subject"]], by_text[triple["object"]]):
            reason = "invalid_entity_type_pair"
        key = (triple["subject"], triple["predicate"], triple["object"])
        if reason:
            rejected.append({"reason": reason, "value": triple})
        elif key not in seen:
            seen.add(key)
            valid.append(triple)
    return valid, rejected


def ollama_request(url: str, model: str, system: str, user: str, timeout: int) -> tuple[list[dict], str, dict]:
    payload = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},
        "keep_alive": "30m",
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    raw = body.get("message", {}).get("content", "")
    parsed = json.loads(raw)
    return parsed.get("relations", []), raw, body


def ollama_draft(url: str, model: str, text: str, timeout: int) -> tuple[list[dict], str, dict]:
    return ollama_request(url, model, RELATION_DRAFT_PROMPT, "文本：\n" + text, timeout)


def ollama_review(
    url: str, model: str, text: str, entities: list[dict], draft: list[dict], timeout: int
) -> tuple[list[dict], str, dict]:
    user = (
        "文本：\n" + text
        + "\n\n实体列表（步骤1输出）：\n" + json.dumps(entities, ensure_ascii=False)
        + "\n\n关系草稿（步骤2输出）：\n" + json.dumps(draft, ensure_ascii=False)
    )
    return ollama_request(url, model, REVIEW_SYSTEM_PROMPT, user, timeout)


def target_capacities(total: int) -> list[int]:
    raw = [total * 0.8, total * 0.1, total * 0.1]
    result = [math.floor(x) for x in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - result[i], reverse=True)[: total - sum(result)]:
        result[index] += 1
    return result


def joint_split(ner_rows: list[dict], relation_by_id: dict[str, list[dict]], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    labels = []
    for row in ner_rows:
        labs = {"ner:" + x["entity_label"] for x in row["output"]}
        labs |= {"rel:" + x["predicate"] for x in relation_by_id[row["text_id"]]}
        labels.append(labs)
    frequency = Counter(label for row_labels in labels for label in row_labels)
    capacities = target_capacities(len(ner_rows))
    desired = {label: target_capacities(count) for label, count in frequency.items()}
    current = {label: [0, 0, 0] for label in frequency}
    parts = [[], [], []]
    order = list(range(len(ner_rows)))
    rng.shuffle(order)
    order.sort(key=lambda i: (sum(1 / frequency[x] for x in labels[i]), len(labels[i])), reverse=True)
    for index in order:
        candidates = [i for i in range(3) if len(parts[i]) < capacities[i]]
        scored = []
        for split in candidates:
            label_need = sum(max(desired[x][split] - current[x][split], 0) / max(desired[x][split], 1) for x in labels[index])
            capacity_need = (capacities[split] - len(parts[split])) / capacities[split]
            scored.append((label_need + 0.15 * capacity_need, capacity_need, rng.random(), split))
        chosen = max(scored)[-1]
        parts[chosen].append(ner_rows[index]["text_id"])
        for label in labels[index]:
            current[label][chosen] += 1
    return dict(zip(("train", "valid", "test"), parts))


def finalize(output_root: Path, ner_rows: list[dict], generated: dict[str, dict], seed: int, model: str) -> dict:
    relation_by_id = {item["text_id"]: item["output"] for item in generated.values()}
    if len(relation_by_id) != len(ner_rows):
        raise ValueError(f"Cannot finalize: generated {len(relation_by_id)}/{len(ner_rows)} rows")
    membership = joint_split(ner_rows, relation_by_id, seed)
    by_id = {row["text_id"]: row for row in ner_rows}
    relation_rows = [
        {"text_id": row["text_id"], "input": row["input"], "output": relation_by_id[row["text_id"]]}
        for row in ner_rows
    ]
    entity_aware_rows = [
        {"text_id": row["text_id"], "input": row["input"], "entities": row["output"], "output": relation_by_id[row["text_id"]]}
        for row in ner_rows
    ]
    write_jsonl(output_root / "ner/entity_all.jsonl", ner_rows)
    write_jsonl(output_root / "relation/relation_all.jsonl", relation_rows)
    write_jsonl(output_root / "relation/relation_all_e.jsonl", entity_aware_rows)
    entity_by_id = {row["text_id"]: row["output"] for row in ner_rows}
    review_rows = [
        {
            "text_id": item["text_id"], "input": item["input"],
            "entities": entity_by_id[item["text_id"]], "source": item["source"],
            "draft_prediction": item.get("draft_prediction", []),
            "accepted_relations": item["output"], "rejected_relations": item.get("rejected", []),
        }
        for item in generated.values() if item.get("rejected")
    ]
    write_jsonl(output_root / "relation_generation/needs_human_review.jsonl", review_rows)
    write_jsonl(output_root / "split_membership.jsonl", [
        {"text_id": item, "split": split} for split, ids in membership.items() for item in ids
    ])
    for split, ids in membership.items():
        wanted = set(ids)
        ner_part = [by_id[x] for x in ids]
        relation_part = [x for x in relation_rows if x["text_id"] in wanted]
        entity_aware_part = [x for x in entity_aware_rows if x["text_id"] in wanted]
        write_jsonl(output_root / f"ner/entity_{split}.jsonl", ner_part)
        write_jsonl(output_root / f"relation/relation_{split}.jsonl", relation_part)
        write_jsonl(output_root / f"relation/relation_{split}_e.jsonl", entity_aware_part)
        # Mirror the unified paper split under benchmark/ for existing CLI compatibility.
        write_jsonl(output_root / f"benchmark/ner_{split}.jsonl", ner_part)
        write_jsonl(output_root / f"benchmark/relation_{split}.jsonl", relation_part)
        write_jsonl(output_root / f"benchmark/relation_{split}_e.jsonl", entity_aware_part)
    source_counts = Counter(item["source"] for item in generated.values())
    report = {
        "model": model,
        "seed": seed,
        "raw_ner_rows": len(read_jsonl(ROOT / "data/ner/entity_all.jsonl")),
        "unique_text_ids": len(ner_rows),
        "relation_rows": len(relation_rows),
        "split_counts": {key: len(value) for key, value in membership.items()},
        "source_counts": dict(source_counts),
        "nonempty_relation_rows": sum(bool(row["output"]) for row in relation_rows),
        "relation_triples": sum(len(row["output"]) for row in relation_rows),
        "rows_needing_review": len(review_rows),
        "human_review_status": "pending" if review_rows else "not_required",
    }
    (output_root / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--output-dir", default=str(ROOT / "data_end/data_generate"))
    parser.add_argument("--ner-source", default=str(ROOT / "data" / "ner" / "entity_all.jsonl"),
                        help="existing NER annotations (text_id/input/output); read entities from here when only annotating relations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Generate at most N unresolved model rows; 0 means all")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    work = output_root / "relation_generation"
    progress_path = work / "progress.jsonl"
    ner_rows = merge_ner(read_jsonl(Path(args.ner_source)))
    old_truth = existing_relation_map()
    completed = {}
    if progress_path.exists():
        for item in read_jsonl(progress_path):
            completed[item["text_id"]] = item

    queued_for_model = 0
    started = time.perf_counter()
    for index, row in enumerate(ner_rows, 1):
        tid, text, entities = row["text_id"], row["input"], row["output"]
        if tid in completed:
            continue
        if text in old_truth:
            valid, rejected = validate_relations(old_truth[text], entities)
            if not rejected:
                item = {"text_id": tid, "input": text, "output": valid, "source": "existing_gold", "rejected": []}
                append_jsonl(progress_path, item)
                completed[tid] = item
                print(f"[{len(completed)}/{len(ner_rows)}] {tid} source={item['source']} relations={len(item['output'])} rejected=0", flush=True)
                continue
            draft = old_truth[text]
            draft_raw = json.dumps({"relations": draft}, ensure_ascii=False)
            source = "existing_gold_repaired"
        else:
            if not has_candidate_pair(entities):
                item = {
                    "text_id": tid, "input": text, "output": [],
                    "source": "schema_validated_negative", "rejected": [],
                }
                append_jsonl(progress_path, item)
                completed[tid] = item
                print(f"[{len(completed)}/{len(ner_rows)}] {tid} source={item['source']} relations=0 rejected=0", flush=True)
                continue
            if args.limit and queued_for_model >= args.limit:
                break
            queued_for_model += 1
            try:
                draft, draft_raw, _ = ollama_draft(args.url, args.model, text, args.timeout)
                source = "ollama_generated_reviewed"
            except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
                draft, draft_raw, source = [], "", "generation_error"
                draft_error = f"{type(exc).__name__}: {exc}"

        if source != "generation_error":
            error = None
            for attempt in range(1, args.retries + 1):
                try:
                    relations, review_raw, metadata = ollama_review(
                        args.url, args.model, text, entities, draft, args.timeout
                    )
                    valid, rejected = validate_relations(relations, entities)
                    if rejected and attempt < args.retries:
                        draft = valid
                        continue
                    item = {
                        "text_id": tid, "input": text, "output": valid, "source": source,
                        "rejected": rejected, "draft_prediction": draft,
                        "raw_draft_prediction": draft_raw, "raw_review_prediction": review_raw,
                        "attempt": attempt, "prompt_eval_count": metadata.get("prompt_eval_count"),
                        "eval_count": metadata.get("eval_count"),
                    }
                    break
                except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(min(2 ** attempt, 8))
            else:
                item = {"text_id": tid, "input": text, "output": [], "source": "generation_error", "rejected": [], "error": error}
        else:
            item = {"text_id": tid, "input": text, "output": [], "source": source, "rejected": [], "error": draft_error}
        append_jsonl(progress_path, item)
        completed[tid] = item
        print(f"[{len(completed)}/{len(ner_rows)}] {tid} source={item['source']} relations={len(item['output'])} rejected={len(item.get('rejected', []))}", flush=True)

    complete = len(completed) == len(ner_rows)
    status = {
        "model": args.model, "completed": len(completed), "total": len(ner_rows), "complete": complete,
        "elapsed_seconds": time.perf_counter() - started,
        "sources": dict(Counter(item["source"] for item in completed.values())),
    }
    work.mkdir(parents=True, exist_ok=True)
    (work / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if complete:
        report = finalize(output_root, ner_rows, completed, args.seed, args.model)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
