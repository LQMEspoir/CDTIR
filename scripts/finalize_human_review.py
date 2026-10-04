"""Record user-attested completion of the current relation truth review."""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

ROOT=Path(os.environ.get("CDTIR_ROOT",Path.cwd())).expanduser().resolve()

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(path,rows):
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reviewer",default="user_attested")
    ap.add_argument("--reviewed-at",default=None)
    args=ap.parse_args(); reviewed_at=args.reviewed_at or datetime.now().astimezone().isoformat(timespec="seconds")
    base=ROOT/"data_end/data_generate"; audit_path=base/"relation_annotation_audit.jsonl"
    rows=read_jsonl(audit_path); changed=0
    for row in rows:
        if row.get("review_status")=="pending":
            row.update({"review_status":"human_verified","reviewer":args.reviewer,"reviewed_at":reviewed_at,
                        "confidence":1.0,"review_reason":"confirmed as final annotation after manual semantic review"})
            changed+=1
    write_jsonl(audit_path,rows)
    review_path=base/"relation_generation"/"needs_human_review.jsonl"
    review_rows=read_jsonl(review_path)
    for row in review_rows:
        row.update({"review_status":"human_verified","reviewer":args.reviewer,"reviewed_at":reviewed_at,
                    "review_reason":"candidate issues reviewed; accepted_relations is the final truth"})
    write_jsonl(review_path,review_rows)
    report_path=base/"generation_report.json"; report=json.loads(report_path.read_text(encoding="utf-8"))
    report.update({"rows_needing_review":0,"rows_human_reviewed":changed,
                   "human_review_status":"completed","reviewer":args.reviewer,"review_completed_at":reviewed_at})
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"human_review_status":"completed","rows_human_reviewed":changed,
                      "review_records_retained":len(review_rows),"reviewed_at":reviewed_at},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
