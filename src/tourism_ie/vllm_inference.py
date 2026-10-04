# File: src/tourism_ie/vllm_inference.py
# Module responsibility: optional vLLM high-throughput inference, replacing per-sample Transformers generate in compatible environments.
# Main data flow: model + a batch of prompts -> vLLM SamplingParams/LLM -> batched generation -> structured predictions.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
import json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from .data_utils import read_jsonl,eval_path,messages_for,is_entity_aware_profile,require_entity_aware_rows
from .parsing import extract_json_payload,normalize_items


def predict_vllm(root: Path,args):
    """- Responsibility: batch-generates a sample set via vLLM and parses output into records consistent with the standard inference path.
    - Parameters: root: Path,args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("vLLM client inference requires openai. Install: pip install -r requirements.txt") from e
    if args.text:
        rows=[{"input":args.text}]
        if getattr(args,"entities_json",None):
            try: rows[0]["entities"]=json.loads(args.entities_json)
            except Exception as e: raise ValueError(f"Invalid --entities-json: {e}") from e
    elif args.input_file:
        rows=[{"input":x.strip()} for x in Path(args.input_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        rows=read_jsonl(eval_path(root,args.task,args.split,args.eval_split,args.prompt_profile))
    if args.limit: rows=rows[:args.limit]
    require_entity_aware_rows(rows,args.prompt_profile,"vLLM inference data")
    output=Path(args.output) if args.output else root/"outputs"/f"{args.task}_vllm_predictions.jsonl"
    output.parent.mkdir(parents=True,exist_ok=True)

    def one(idx,row):
        """- Responsibility: runs the local "one" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: idx,row.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: may read/write disk files/model weights; ensure the output directory is writable.
        """
        client=OpenAI(api_key=args.api_key,base_url=args.base_url)
        start=time.perf_counter()
        resp=client.chat.completions.create(
            model=args.served_model,messages=messages_for(args.task,row["input"],args.prompt_profile,row.get("entities",[]) if args.task=="relation" and is_entity_aware_profile(args.prompt_profile) else None),
            temperature=args.temperature,max_tokens=args.max_new_tokens,
        )
        elapsed=time.perf_counter()-start
        raw=resp.choices[0].message.content or ""
        rec={"input":row["input"],"prediction":normalize_items(args.task,extract_json_payload(raw)),
             "raw_prediction":raw,"inference_seconds":elapsed,"prompt_profile":args.prompt_profile,
             "served_model":args.served_model}
        if "output" in row: rec["gold"]=normalize_items(args.task,row["output"])
        return idx,rec

    results=[None]*len(rows); start_all=time.perf_counter()
    workers=max(1,int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(one,i,r) for i,r in enumerate(rows)]
        done=0
        for fut in as_completed(futs):
            i,rec=fut.result(); results[i]=rec; done+=1
            print(f"[{done}/{len(rows)}] {rec['input'][:60]} -> {json.dumps(rec['prediction'],ensure_ascii=False)}")
    wall=time.perf_counter()-start_all
    with output.open("w",encoding="utf-8",newline="\n") as f:
        for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    summed=sum(r["inference_seconds"] for r in results)
    info={"output":str(output),"samples":len(results),"wall_seconds":wall,"summed_request_seconds":summed,
          "wall_seconds_per_sample":wall/max(1,len(results)),"workers":workers,"base_url":args.base_url,"served_model":args.served_model}
    print(json.dumps(info,ensure_ascii=False,indent=2)); return info
