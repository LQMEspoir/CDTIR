"""Build NER + relation truth from raw crawled text with Ollama.

This is the NER-supplemented version of generate_relation_truth_ollama.py: the original only does relation extraction
(NER reuses existing annotations in data/ner); this script also brings the paper §4.1.3 first step (NER entity recognition)
into the same multi-role + schema-validation flow, using Ollama to generate entities on the fly from raw unlabeled text,
then run the original three relation steps (draft -> review -> schema validation), and finally reuse finalize to rebuild the whole tree.

Relation to annotate_data.py:
  - annotate_data.py loads Qwen2.5-7B locally via transformers;
  - this script goes through the local Ollama API (identical to main2's generate_relation_truth_ollama.py).

The NER prompt uses a flat 15-class list (not the paper Table 1 group headers), and uses Ollama's format: JSON Schema
to lock entity_label to the 15 classes via enum, avoiding group-name leakage (e.g., Location-related) from the start.

Progress is append-only and resumable: progress rows also store entities, so reruns don't need to re-label NER.
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


ROOT = Path(os.environ.get("CDTIR_ROOT", Path.cwd())).expanduser().resolve()

NER_LABELS = ["目的地", "出发地", "返程地", "餐饮", "住宿", "产品", "交通", "预算",
              "时长", "时间", "人群", "强度", "人气", "人数", "天气"]
RELATIONS = ["游玩顺序", "出行时间", "停留时长", "预算限制", "人均预算", "体验内容"]
ALLOWED = {
    "体验内容": ({"目的地", "餐饮", "住宿"}, {"产品"}),
    "预算限制": ({"目的地", "餐饮", "住宿", "产品", "交通"}, {"预算"}),
    "人均预算": ({"预算"}, {"人数"}),
    "停留时长": ({"目的地", "餐饮", "住宿", "产品"}, {"时长"}),
    "出行时间": ({"出发地", "返程地", "目的地", "餐饮", "住宿", "产品"}, {"时间"}),
}
ORDER_PAIRS = {("出发地", "目的地"), ("目的地", "目的地"), ("目的地", "返程地")}

# relation extraction JSON schema (kept from the original script)
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

# NER JSON schema: entity_label locked to 15 classes via enum
NER_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_text": {"type": "string"},
                    "entity_label": {"type": "string", "enum": NER_LABELS},
                },
                "required": ["entity_text", "entity_label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}

# NER uses a flat-list prompt (consistent with original entity_router_rola.py, to avoid group headers leaking as labels),
# and adds few-shot examples + emphasizes "scenic/place names are always Destination" to boost zero-shot recall.
# Chinese NER annotation prompt. English translation:
#   "You are an expert in text entity recognition; extract the following 15 entity types from the given sentence:
#    Destination; Origin; Return; Dining; Lodging; Product; Transport; Budget; Duration; Time; Crowd; Intensity; Popularity; People; Weather.
#    entity_label must be one of the 15 types. Recognize all entities (attraction/place/hotel/food/time/days/people/budget); attraction/place names are labeled "Destination".
#    Examples (3 Chinese text -> entity examples below). Output an empty array only when no entity is found."
NER_SYSTEM_PROMPT = """你是一个文本实体识别领域的专家，请从给定的句子中提取以下 15 类实体：
目的地; 出发地; 返程地; 餐饮; 住宿; 产品; 交通; 预算; 时长; 时间; 人群; 强度; 人气; 人数; 天气。

entity_label 只能是上述 15 类之一。请尽量完整地识别文本中提到的所有实体：景点名、地名、酒店名、美食名、时间、天数、人数、预算等都要标出；景点名/地名一律标为「目的地」。

示例1：
文本：台州府城文化旅游区怎么样？好玩吗？
实体：[{"entity_text":"台州府城文化旅游区","entity_label":"目的地"}]

示例2：
文本：从北京出发去海拉尔玩3天，求路线
实体：[{"entity_text":"北京","entity_label":"出发地"},{"entity_text":"海拉尔","entity_label":"目的地"},{"entity_text":"3天","entity_label":"时长"}]

示例3：
文本：丽江古城附近有酒店和美食推荐吗
实体：[{"entity_text":"丽江古城","entity_label":"目的地"},{"entity_text":"酒店","entity_label":"住宿"},{"entity_text":"美食","entity_label":"餐饮"}]

找不到任何实体时才输出空数组。"""

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
- 体验内容：(目的地/餐饮/住宿, 产品)
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


def normalize_entities(entities: list) -> list[dict]:
    """Filter non-15-class labels, drop empty, dedup by (text, label) (fallback for Ollama output)."""
    out, seen = [], set()
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        text = str(e.get("entity_text", "")).strip()
        label = str(e.get("entity_label", "")).strip()
        if text and label in NER_LABELS and (text, label) not in seen:
            seen.add((text, label))
            out.append({"entity_text": text, "entity_label": label})
    return out


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
    # reuse existing gold from data_end/data_generate (paper tree, 1011 rows), not data/relation (old tree, 399 rows),
    # otherwise the few overlapping rows between newly crawled text and old data won't be reused and would be regenerated inconsistently.
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


def ollama_request(url: str, model: str, system: str, user: str, timeout: int, schema: dict = SCHEMA):
    payload = {
        "model": model,
        "stream": False,
        "format": schema,
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
    return parsed, raw, body


def ollama_draft(url: str, model: str, text: str, timeout: int) -> tuple[list[dict], str, dict]:
    parsed, raw, body = ollama_request(url, model, RELATION_DRAFT_PROMPT, "文本：\n" + text, timeout)
    return parsed.get("relations", []), raw, body


def ollama_review(url: str, model: str, text: str, entities: list[dict], draft: list[dict], timeout: int):
    user = (
        "文本：\n" + text
        + "\n\n实体列表（步骤1输出）：\n" + json.dumps(entities, ensure_ascii=False)
        + "\n\n关系草稿（步骤2输出）：\n" + json.dumps(draft, ensure_ascii=False)
    )
    parsed, raw, body = ollama_request(url, model, REVIEW_SYSTEM_PROMPT, user, timeout)
    return parsed.get("relations", []), raw, body


def ollama_ner(url: str, model: str, text: str, timeout: int) -> tuple[list[dict], str, dict]:
    parsed, raw, body = ollama_request(url, model, NER_SYSTEM_PROMPT, "文本：\n" + text, timeout, NER_SCHEMA)
    return normalize_entities(parsed.get("entities", [])), raw, body


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
        write_jsonl(output_root / f"benchmark/ner_{split}.jsonl", ner_part)
        write_jsonl(output_root / f"benchmark/relation_{split}.jsonl", relation_part)
        write_jsonl(output_root / f"benchmark/relation_{split}_e.jsonl", entity_aware_part)
    source_counts = Counter(item["source"] for item in generated.values())
    report = {
        "model": model,
        "seed": seed,
        "ner_source_rows": len(ner_rows),
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
    parser.add_argument("--ner-source", default=str(ROOT / "data_end/data_more" / "cleaned_intent.jsonl"),
                        help="unlabeled raw text JSONL (field: input), used to supplement NER annotations")
    parser.add_argument("--output-dir", default=str(ROOT / "data_end/data_expanded"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Generate at most N unresolved model rows; 0 means all")
    parser.add_argument("--reannotate", action="store_true",
                        help="ignore existing progress, re-annotate all NER + relations (use after changing the NER prompt)")
    parser.add_argument("--reannotate-relation", action="store_true",
                        help="reuse NER entities from existing progress, rerun relation draft+review only (use after changing the relation prompt)")
    parser.add_argument("--entities-source", default=None,
                        help="read entities from the human-reviewed NER file (text_id + entities/output); skip NER and annotate relations directly")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    work = output_root / "relation_generation"
    progress_path = work / "progress.jsonl"

    # read raw unlabeled text, dedup and assign text_id by the main2 convention (with the "文本:" ("text:") prefix)
    source: list[str] = []
    seen: set[str] = set()
    for r in read_jsonl(Path(args.ner_source)):
        text = str(r.get("input", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        source.append(text)

    old_truth = existing_relation_map()
    completed: dict[str, dict] = {}
    old_entities: dict[str, list] = {}
    if args.reannotate and progress_path.exists():
        progress_path.rename(progress_path.with_suffix(progress_path.suffix + ".bak"))
        print(f"reannotate: backed up old progress -> {progress_path.name}.bak, full re-label")
    elif args.entities_source and progress_path.exists():
        progress_path.rename(progress_path.with_suffix(progress_path.suffix + ".bak"))
        print(f"entities-source: backed up old progress, re-label relations with external entities")
    elif args.reannotate_relation and progress_path.exists():
        for item in read_jsonl(progress_path):
            old_entities[item["text_id"]] = item.get("entities", [])
        progress_path.rename(progress_path.with_suffix(progress_path.suffix + ".bak"))
        print(f"reannotate-relation: backed up old progress, reuse {len(old_entities)} rows NER entities, rerun relations only")
    if progress_path.exists():
        for item in read_jsonl(progress_path):
            completed[item["text_id"]] = item

    # after human entity review: read entities straight from the NER file, skip NER, only annotate relations
    entities_by_id: dict[str, list] = {}
    if args.entities_source:
        for r in read_jsonl(Path(args.entities_source)):
            ents = r.get("entities", r.get("output", []))
            if ents:
                entities_by_id[r["text_id"]] = ents
        print(f"from {args.entities_source} read {len(entities_by_id)}  rows entities, skip NER to annotate relations directly")

    queued_for_model = 0
    started = time.perf_counter()
    for text in source:
        tid = text_id(text)
        if tid in completed:
            continue

        # step 1: NER (complete entities, with retry); reuse existing entities when --entities-source / --reannotate-relation is set
        if args.entities_source:
            entities = entities_by_id.get(tid, [])
            ner_error = None
        elif args.reannotate_relation:
            entities = old_entities.get(tid, [])
            ner_error = None
        else:
            entities: list[dict] = []
            ner_error = None
            for attempt in range(1, args.retries + 1):
                try:
                    entities, _, _ = ollama_ner(args.url, args.model, text, args.timeout)
                    ner_error = None
                    break
                except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
                    ner_error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(min(2 ** attempt, 8))

        # steps 2+3: relations (reuse original logic; entities now come from Ollama NER)
        if text in old_truth:
            valid, rejected = validate_relations(old_truth[text], entities)
            if not rejected:
                item = {"text_id": tid, "input": text, "entities": entities, "output": valid,
                        "source": "existing_gold", "rejected": []}
                append_jsonl(progress_path, item)
                completed[tid] = item
                print(f"[{len(completed)}/{len(source)}] {tid} source={item['source']} entities={len(entities)} relations={len(item['output'])} rejected=0", flush=True)
                continue
            draft = old_truth[text]
            draft_raw = json.dumps({"relations": draft}, ensure_ascii=False)
            source_label = "existing_gold_repaired"
        else:
            if ner_error:
                item = {"text_id": tid, "input": text, "entities": [], "output": [],
                        "source": "ner_error", "rejected": [], "error": ner_error}
                append_jsonl(progress_path, item)
                completed[tid] = item
                print(f"[{len(completed)}/{len(source)}] {tid} source={item['source']} error={ner_error}", flush=True)
                continue
            if not has_candidate_pair(entities):
                item = {"text_id": tid, "input": text, "entities": entities, "output": [],
                        "source": "schema_validated_negative", "rejected": []}
                append_jsonl(progress_path, item)
                completed[tid] = item
                print(f"[{len(completed)}/{len(source)}] {tid} source={item['source']} entities={len(entities)} relations=0 rejected=0", flush=True)
                continue
            if args.limit and queued_for_model >= args.limit:
                break
            queued_for_model += 1
            try:
                draft, draft_raw, _ = ollama_draft(args.url, args.model, text, args.timeout)
                source_label = "ollama_generated_reviewed"
            except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
                draft, draft_raw, source_label = [], "", "generation_error"
                draft_error = f"{type(exc).__name__}: {exc}"

        if source_label != "generation_error":
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
                        "text_id": tid, "input": text, "entities": entities, "output": valid, "source": source_label,
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
                item = {"text_id": tid, "input": text, "entities": entities, "output": [],
                        "source": "generation_error", "rejected": [], "error": error}
        else:
            item = {"text_id": tid, "input": text, "entities": entities, "output": [],
                    "source": source_label, "rejected": [], "error": draft_error}
        append_jsonl(progress_path, item)
        completed[tid] = item
        print(f"[{len(completed)}/{len(source)}] {tid} source={item['source']} entities={len(entities)} relations={len(item['output'])} rejected={len(item.get('rejected', []))}", flush=True)

    complete = len(completed) == len(source)
    status = {
        "model": args.model, "completed": len(completed), "total": len(source), "complete": complete,
        "elapsed_seconds": time.perf_counter() - started,
        "sources": dict(Counter(item["source"] for item in completed.values())),
    }
    work.mkdir(parents=True, exist_ok=True)
    (work / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if complete:
        ner_rows = [{"text_id": tid, "input": item["input"], "output": item.get("entities", [])}
                    for tid, item in completed.items()]
        report = finalize(output_root, ner_rows, completed, args.seed, args.model)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
