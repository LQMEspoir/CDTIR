# File: src/tourism_ie/original_data.py
# Module responsibility: data conversion and compatibility layer for the original project: normalizes legacy Swift/raw JSONL into the current NER/NRE format and generates entity-aware RE data.
# Main data flow: raw reference/source files -> parse user text/entities/assistant output -> canonical rows -> split by original membership and write standard data.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
import json, re
from pathlib import Path
from .parsing import extract_json_payload, normalize_items
from .data_utils import read_jsonl


def _read_source(path: Path):
    """- Responsibility: reads original reference/source files, compatible with JSON, JSONL or legacy export formats.
    - Parameters: path: Path.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;includes explicit parameter/data validation; raises on invalid input.
    """
    rows=[]
    with path.open(encoding="utf-8") as f:
        for no,line in enumerate(f,1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except Exception as e: raise ValueError(f"Invalid original source {path}:{no}: {e}") from e
    return rows


def _canonical(task: str,row: dict):
    """- Responsibility: normalizes legacy records into the current unified field structure, reducing branching for multiple old schemas.
    - Parameters: task: str,row: dict.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    value=row.get("output",[])
    if isinstance(value,str): value=extract_json_payload(value)
    return {"instruction":row.get("instruction","") if task=="ner" else row.get("instruction",""),
            "input":row["input"],"output":normalize_items(task,value)}


def _write_jsonl(path: Path,rows):
    """- Responsibility: writes a dict sequence to disk as UTF-8 JSONL, ensuring the parent dir exists.
    - Parameters: path: Path,rows.
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")


def _membership(root: Path,task: str):
    """- Responsibility: builds a text-to-split mapping from the original train/test or valid membership.
    - Parameters: root: Path,task: str.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    if task=="ner":
        a=read_jsonl(root/"data_end/data/ner/entity_train.jsonl"); b=read_jsonl(root/"data_end/data/ner/entity_test.jsonl")
        return {"train":{x["input"] for x in a},"test":{x["input"] for x in b}}
    a=read_jsonl(root/"data_end/data/relation/relation_train.jsonl"); b=read_jsonl(root/"data_end/data/relation/relation_valid.jsonl")
    return {"train":{x["input"] for x in a},"valid":{x["input"] for x in b}}


def prepare_original_data(root: Path,args):
    """- Responsibility: rebuilds the current NER/NRE standard files from packaged original reference data, preserving original set membership.
    - Parameters: root: Path,args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable.
    """
    """Canonicalize the immutable original source files shipped in data/original_reference.

    Split membership is preserved from the package's active original split, so this
    command rebuilds the same train/test (NER) and train/valid (NRE) flow rather than
    silently inventing a new random partition.
    """
    ner_source=Path(args.ner_source) if args.ner_source else root/"data_end/data/original_reference/entity_output_original.jsonl"
    rel_source=Path(args.relation_source) if args.relation_source else root/"data_end/data/original_reference/relationship_output_type_original.jsonl"
    out=Path(args.output_dir) if args.output_dir else root/"data_end/data/rebuilt_original"
    ner=[_canonical("ner",r) for r in _read_source(ner_source)]
    rel=[_canonical("relation",r) for r in _read_source(rel_source)]
    # Remove empty instruction from relation canonical rows to keep the active schema concise.
    rel=[{"input":r["input"],"output":r["output"]} for r in rel]
    nm=_membership(root,"ner"); rm=_membership(root,"relation")
    ner_train=[r for r in ner if r["input"] in nm["train"]]; ner_test=[r for r in ner if r["input"] in nm["test"]]
    # One legacy source text was later high-confidence corrected from "日照701..." (Rizhao) to "黄山701..." (Huangshan).
    # Preserve the original 90/10 split cardinality by placing source-only unmatched rows
    # in the held-out side once the 911-row train membership has been recovered.
    assigned_ner={r["input"] for r in ner_train+ner_test}
    ner_unassigned=[r for r in ner if r["input"] not in assigned_ner]
    desired_train=round(len(ner)*0.9)
    for r in ner_unassigned:
        (ner_train if len(ner_train)<desired_train else ner_test).append(r)
    rel_train=[r for r in rel if r["input"] in rm["train"]]; rel_valid=[r for r in rel if r["input"] in rm["valid"]]
    files={
        "ner_all":(out/"ner_all.jsonl",ner),"ner_train":(out/"ner_train.jsonl",ner_train),"ner_test":(out/"ner_test.jsonl",ner_test),
        "relation_all":(out/"relation_all.jsonl",rel),"relation_train":(out/"relation_train.jsonl",rel_train),"relation_valid":(out/"relation_valid.jsonl",rel_valid),
    }
    for _,(p,rows) in files.items(): _write_jsonl(p,rows)
    report={"ner_source":str(ner_source),"relation_source":str(rel_source),"output_dir":str(out),
            "counts":{k:len(rows) for k,(_,rows) in files.items()},
            "unassigned_ner":len(ner)-len(ner_train)-len(ner_test),"unassigned_relation":len(rel)-len(rel_train)-len(rel_valid)}
    (out/"rebuild_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


# ── Entity-aware RE data conversion ──────────────────────────────────────────

def _split_text_entities(raw: str) -> tuple[str, list[dict]]:
    """- Responsibility: splits the raw sentence and the entity-list part from a legacy entity-aware text.
    - Parameters: raw: str.
    - Returns: return type is `tuple[str, list[dict]]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Parse ``文本:...\n实体:{...}{...}`` while preserving the single 文本: prefix."""
    raw=(raw or "").strip()
    if "\n实体:" in raw:
        # if "\nentities:" in raw:
        text, entity_str = raw.split("\n实体:",1)
        # text, entity_str = raw.split("\nentities:",1)
    else:
        text, entity_str = raw, ""
    text=text.strip()
    if text and not text.startswith("文本:"):
        # if text and not text.startswith("text:"):
        text=f"文本:{text}"
        # text=f"text:{text}"
    entities=[]
    decoder=json.JSONDecoder()
    pos=0
    entity_str=entity_str.strip()
    while pos < len(entity_str):
        while pos < len(entity_str) and entity_str[pos].isspace(): pos += 1
        if pos >= len(entity_str): break
        try:
            obj,end=decoder.raw_decode(entity_str,pos)
        except json.JSONDecodeError:
            # Keep conversion robust to historical malformed fragments.
            m=re.search(r'\{[^}]+\}',entity_str[pos:])
            if not m: break
            try: obj=json.loads(m.group())
            except Exception:
                pos += max(1,m.end()); continue
            end=pos+m.end()
        if isinstance(obj,dict) and obj.get("entity_text") and obj.get("entity_label"):
            entities.append({"entity_text":str(obj["entity_text"]),"entity_label":str(obj["entity_label"])})
        pos=end
    return text,entities


def _parse_swift_user_content(user_content: str) -> tuple[str, list[dict]]:
    """- Responsibility: parses user content in Swift-style chat records and extracts the task input.
    - Parameters: user_content: str.
    - Returns: return type is `tuple[str, list[dict]]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Extract the final original E-branch text/entity payload from a Swift user turn."""
    # train_e user messages contain the full instruction followed by the payload.
    # Take the final 文本: ("text:") occurrence, not an earlier occurrence in explanatory prose.
    pos=user_content.rfind("文本:")
    # pos=user_content.rfind("text:")
    payload=user_content[pos:] if pos >= 0 else user_content
    return _split_text_entities(payload)


def _parse_assistant_content(assistant_content: str) -> list[dict]:
    """- Responsibility: parses Swift-style assistant output and restores strings into structured entity/relation annotations.
    - Parameters: assistant_content: str.
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    payload = extract_json_payload(assistant_content)
    return normalize_items("relation", payload)


def convert_entity_aware_re(source: Path, output: Path) -> dict:
    """- Responsibility: converts legacy text+entity relation data into the current unified entity-aware RE row format.
    - Parameters: source: Path,output: Path.
    - Returns: return type is `dict`; see implementation below for field details.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;includes explicit parameter/data validation; raises on invalid input.
    """
    """Convert original train_e.json / valid_e.json (Swift messages) to entity-aware JSONL."""
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    rows=[]
    for item in data:
        msgs=item.get("messages",[])
        if len(msgs)<2: continue
        text,entities=_parse_swift_user_content(msgs[0].get("content",""))
        triples=_parse_assistant_content(msgs[1].get("content",""))
        if not text: continue
        rows.append({"input":text,"entities":entities,"output":triples})
    _write_jsonl(output,rows)
    return {"source":str(source),"output":str(output),"rows":len(rows),
            "rows_with_entities":sum(bool(r.get("entities")) for r in rows)}


def _packaged_entity_aware_rows(source: Path) -> list[dict]:
    """- Responsibility: reads and normalizes entity-aware relation samples from packaged project resources.
    - Parameters: source: Path.
    - Returns: return type is `list[dict]`; see implementation below for field details.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Build E-branch rows from packaged relationship_output_entity.jsonl."""
    rows=[]
    for raw in _read_source(source):
        text,entities=_split_text_entities(raw.get("input",""))
        triples=normalize_items("relation",raw.get("output",[]))
        if not text: continue
        rows.append({"input":text,"entities":entities,"output":triples})
    return rows


def _split_entity_rows_by_membership(root: Path, rows: list[dict], split_scheme: str) -> dict[str,list[dict]]:
    """- Responsibility: assigns entity-aware samples back to train/valid/test by original split membership.
    - Parameters: root: Path,rows: list[dict],split_scheme: str.
    - Returns: return type is `dict[str, list[dict]]`; see implementation below for field details.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    by_text={r["input"]:r for r in rows}
    if split_scheme=="original":
        sources={
            "train":root/"data_end/data/relation/relation_train.jsonl",
            "valid":root/"data_end/data/relation/relation_valid.jsonl",
        }
    else:
        sources={
            "train":root/"data_end/data/benchmark/relation_train.jsonl",
            "valid":root/"data_end/data/benchmark/relation_valid.jsonl",
            "test":root/"data_end/data/benchmark/relation_test.jsonl",
        }
    result={}
    missing=[]
    for split,path in sources.items():
        selected=[]
        for x in read_jsonl(path):
            r=by_text.get(x["input"])
            if r is None: missing.append(x["input"])
            else: selected.append(r)
        result[split]=selected
    if missing:
        raise ValueError(f"Entity-aware source is missing {len(missing)} texts required by {split_scheme} split; first={missing[0]!r}")
    return result


def prepare_entity_aware_data(root: Path, args=None) -> dict:
    """- Responsibility: generates and saves entity-aware NRE data files for strict text-only vs entity-aware comparison.
    - Parameters: root: Path,args=None.
    - Returns: return type is `dict`; see implementation below for field details.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;includes explicit parameter/data validation; raises on invalid input.
    """
    """Prepare runnable E-branch RE data.

    Priority:
    1) explicit/original train_e.json + valid_e.json Swift files when available;
    2) packaged data/supplementary/relationship_output_entity.jsonl (399 rows).

    In both cases original 319/80 and benchmark 318/41/40 memberships are preserved.
    """
    source_dir=Path(args.source_dir) if args is not None and getattr(args,"source_dir",None) else root/"data_end/data/original_reference"
    out_dir=Path(args.output_dir) if args is not None and getattr(args,"output_dir",None) else root/"data_end/data/relation"
    train_swift=source_dir/"train_e.json"; valid_swift=source_dir/"valid_e.json"
    mode="swift"
    if train_swift.exists() and valid_swift.exists():
        tmp_train=out_dir/"relation_train_e.jsonl"; tmp_valid=out_dir/"relation_valid_e.jsonl"
        tr=convert_entity_aware_re(train_swift,tmp_train); va=convert_entity_aware_re(valid_swift,tmp_valid)
        all_rows=read_jsonl(tmp_train)+read_jsonl(tmp_valid)
        source_desc=f"{train_swift};{valid_swift}"
    else:
        mode="packaged_supplementary"
        source=root/"data_end/data/supplementary/relationship_output_entity.jsonl"
        if not source.exists():
            raise FileNotFoundError(
                f"Entity-aware sources not found. Expected {train_swift} + {valid_swift}, "
                f"or packaged fallback {source}."
            )
        all_rows=_packaged_entity_aware_rows(source)
        source_desc=str(source)

    original=_split_entity_rows_by_membership(root,all_rows,"original")
    for split,part in original.items():
        _write_jsonl(out_dir/f"relation_{split}_e.jsonl",part)

    benchmark=_split_entity_rows_by_membership(root,all_rows,"benchmark")
    bench_dir=root/"data_end/data/benchmark"
    for split,part in benchmark.items():
        _write_jsonl(bench_dir/f"relation_{split}_e.jsonl",part)

    report={
        "mode":mode,"source":source_desc,"all_rows":len(all_rows),
        "rows_with_entities":sum(bool(r.get("entities")) for r in all_rows),
        "original_counts":{k:len(v) for k,v in original.items()},
        "benchmark_counts":{k:len(v) for k,v in benchmark.items()},
        "output_dir":str(out_dir),"benchmark_output_dir":str(bench_dir),
    }
    report_path=out_dir/"entity_aware_prepare_report.json"
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
