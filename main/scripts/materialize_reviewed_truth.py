"""Materialize the reviewed relation audit back into canonical JSONL files."""
from __future__ import annotations
import argparse, hashlib, json
import os
from datetime import datetime
from pathlib import Path

ROOT=Path(os.environ.get("CDTIR_ROOT",Path.cwd())).expanduser().resolve()

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="data_end/data_generate"); ap.add_argument("--reviewed-at",default=None)
    args=ap.parse_args(); base=ROOT/args.data_dir
    audit=read_jsonl(base/"relation_annotation_audit.jsonl")
    if len(audit)!=1011 or len({x["text_id"] for x in audit})!=1011: raise SystemExit("relation audit must contain 1011 unique text_id rows")
    pending=[x for x in audit if x.get("review_status") not in {"human_verified","rule_or_existing"}]
    if pending: raise SystemExit(f"cannot materialize: {len(pending)} audit rows are not finalized")
    by_id={x["text_id"]:x for x in audit}; all_path=base/"relation"/"relation_all.jsonl"; rows=read_jsonl(all_path)
    if len(rows)!=1011: raise SystemExit(f"unexpected relation_all rows: {len(rows)}")
    changed=0
    for row in rows:
        final=by_id[row["text_id"]]["new_value"]
        if row.get("output")!=final: changed+=1
        row["output"]=final
    write_jsonl(all_path,rows); relation_by_id={x["text_id"]:x for x in rows}
    members=read_jsonl(base/"split_membership.jsonl")
    for split in ("train","valid","test"):
        ids=[x["text_id"] for x in members if x["split"]==split]
        ner=read_jsonl(base/"ner"/f"entity_{split}.jsonl"); relation=[relation_by_id[i] for i in ids]
        e=[{"text_id":n["text_id"],"input":n["input"],"entities":n.get("output",[]),"output":relation_by_id[n["text_id"]]["output"]} for n in ner]
        for folder,name,value in (("relation",f"relation_{split}.jsonl",relation),("relation",f"relation_{split}_e.jsonl",e),("benchmark",f"relation_{split}.jsonl",relation),("benchmark",f"relation_{split}_e.jsonl",e)):
            write_jsonl(base/folder/name,value)
    final_path=base/"relation_truth_final.jsonl"; write_jsonl(final_path,rows)
    h=hashlib.sha256((base/"relation_annotation_audit.jsonl").read_bytes()).hexdigest()
    report={"status":"materialized","rows":len(rows),"rewritten_rows":changed,"triples":sum(len(x["output"]) for x in rows),
            "human_verified_rows":sum(x.get("review_status")=="human_verified" for x in audit),
            "reviewed_at":args.reviewed_at or datetime.now().astimezone().isoformat(timespec="seconds"),"audit_sha256":h,
            "final_file":str(final_path)}
    (base/"final_truth_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
