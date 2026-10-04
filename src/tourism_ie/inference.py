# File: src/tourism_ie/inference.py
# Module responsibility: Transformers/PEFT inference: single or dataset prediction with a base model, plain adapter, or Router-MultiLoRA package.
# Main data flow: model/adapter/router config + text or eval split -> build prompt -> generate -> parse structured output -> save prediction JSONL.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
from pathlib import Path
import json, time, torch
from .data_utils import read_jsonl, eval_path, messages_for, render_prompt, is_entity_aware_profile, require_entity_aware_rows
from .modeling import load_tokenizer, load_base_model
from .parsing import extract_json_payload, normalize_items


def _load(model_name: str, adapter: str | None, quantized: bool = False):
    """- Responsibility: loads tokenizer, base model and optional PEFT adapter, set to an inference-ready state.
    - Parameters: model_name: str,adapter: str | None,quantized: bool=False.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: involves training/backprop; memory/randomness affect resource usage and reproducibility;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input.
    """
    tok=load_tokenizer(model_name)
    model=load_base_model(model_name,training=False,quantized=quantized)
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise SystemExit("Adapter inference requires peft. Run pip install -r requirements.txt") from e
        model=PeftModel.from_pretrained(model,adapter)
    if not getattr(model,"hf_device_map",None):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        try: device=next(model.parameters()).device
        except StopIteration: device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok,model,device


def _generate(task,text,tok,model,device,max_new_tokens,prompt_profile="canonical",entities=None):
    """- Responsibility: tokenizes the built prompt, runs model.generate, and slices/decodes the newly generated tokens.
    - Parameters: task,text,tok,model,device,max_new_tokens,prompt_profile='canonical',entities=None.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: inference/tensor paths must keep device and dtype consistent.
    """
    prompt=render_prompt(tok,task,text,prompt_profile,add_generation_prompt=True,entities=entities)
    inputs=tok(prompt,return_tensors="pt")
    inputs={k:v.to(device) for k,v in inputs.items()}
    start=time.perf_counter()
    with torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                           pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    elapsed=time.perf_counter()-start
    gen=ids[0,inputs["input_ids"].shape[1]:]
    raw=tok.decode(gen,skip_special_tokens=True).strip()
    return raw, normalize_items(task, extract_json_payload(raw)), elapsed


def _router_format(router_dir: str|Path):
    """- Responsibility: converts Router-MultiLoRA output into the same structured format as the plain prediction path.
    - Parameters: router_dir: str | Path.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    p=Path(router_dir)/"router_manifest.json"
    manifest=json.loads(p.read_text(encoding="utf-8"))
    return manifest.get("format",""),manifest


def predict(root: Path,args):
    """- Responsibility: unified prediction flow: load model/adapter or router package, build prompt, generate, parse structured result, save to file.
    - Parameters: root: Path,args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Key process: distinguish router-package / plain-adapter / base-model paths; build a unified prompt; parse via the parsing module into structured entities/relations.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    router_dir=getattr(args,"router_dir",None)
    router_threshold=getattr(args,"router_threshold",None)
    router_kind=None
    if router_dir:
        fmt,_=_router_format(router_dir)
        if str(fmt).startswith("tourism_ie_neural_router_multilora"):
            from .router_multilora import load_router_package
            tok,model,router,manifest,device=load_router_package(router_dir,getattr(args,"load_in_4bit",False))
            router_kind="neural"
        elif str(fmt).startswith("tourism_ie_grouped_router_multilora") or fmt=="tourism_ie_router_multilora_v1":
            from .router_grouped import load_router_package
            tok,model,router,manifest,device=load_router_package(router_dir,getattr(args,"load_in_4bit",False))
            router_kind="grouped"
        else:
            raise ValueError(f"Unknown router package format: {fmt!r}")
        if manifest.get("task") != args.task:
            raise ValueError(f"Router package task={manifest.get('task')!r} does not match requested task={args.task!r}")
    else:
        tok,model,device=_load(args.model,args.adapter,getattr(args,"load_in_4bit",False))
        router=manifest=None

    # Router packages own their training prompt profile; use it to choose the matching
    # text-only vs entity-aware evaluation file before reading rows.
    data_prompt_profile=(manifest.get("prompt_profile",getattr(args,"prompt_profile","canonical"))
                         if router_kind else getattr(args,"prompt_profile","canonical"))
    rows=[]
    if args.text:
        rows=[{"input":args.text}]
        if getattr(args,"entities_json",None):
            try: rows[0]["entities"]=json.loads(args.entities_json)
            except Exception as e: raise ValueError(f"Invalid --entities-json: {e}") from e
    elif getattr(args,"dataset_file",None):
        rows=read_jsonl(Path(args.dataset_file))
    elif args.input_file:
        rows=[{"input":x.strip()} for x in Path(args.input_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        path=eval_path(root,args.task,getattr(args,"split","original"),getattr(args,"eval_split","test"),data_prompt_profile,getattr(args,"data_dir",None))
        rows=read_jsonl(path)
    if args.limit: rows=rows[:args.limit]
    require_entity_aware_rows(rows,data_prompt_profile,"inference data")
    output=Path(args.output) if args.output else root/"outputs"/f"{args.task}_predictions.jsonl"
    output.parent.mkdir(parents=True,exist_ok=True)
    total_seconds=0.0
    route_counts={}
    saved_records=[]
    with output.open("w",encoding="utf-8",newline="\n") as f:
        for i,row in enumerate(rows,1):
            route_name=route_confidence=None; route_weights=None
            prompt_profile=getattr(args,"prompt_profile","canonical")
            if router_kind:
                prompt_profile=manifest.get("prompt_profile",prompt_profile)
            # E-branch: extract entities from row if available
            entities = row.get("entities", []) if args.task == "relation" and is_entity_aware_profile(prompt_profile) else None
            if router_kind=="neural" and getattr(args,"router_generation_mode","top1")=="weighted":
                from .router_multilora import generate_weighted
                raw,seconds,route_weights=generate_weighted(
                    args.task,row["input"],tok,model,router,manifest,device,args.max_new_tokens,
                    getattr(args,"router_generation_top_k",0),entities=entities)
                parsed=normalize_items(args.task,extract_json_payload(raw))
                if route_weights:
                    route_name=max(route_weights,key=route_weights.get); route_confidence=route_weights[route_name]
            else:
                if router_kind=="neural":
                    from .router_multilora import route_text
                    route_name,route_confidence,route_weights=route_text(row["input"],tok,model,router,manifest,device,router_threshold,entities=entities)
                    model.set_adapter(route_name)
                elif router_kind=="grouped":
                    from .router_grouped import route_text
                    route_name,route_confidence=route_text(row["input"],router,manifest,router_threshold)
                    model.set_adapter(route_name)
                raw,parsed,seconds=_generate(args.task,row["input"],tok,model,device,args.max_new_tokens,prompt_profile,entities=entities)
            total_seconds+=seconds
            rec={"input":row["input"],"prediction":parsed,"raw_prediction":raw,"inference_seconds":seconds,"prompt_profile":prompt_profile}
            if "text_id" in row: rec["text_id"]=row["text_id"]
            if "entities" in row: rec["input_entities"]=row["entities"]
            if route_name is not None:
                rec["router_adapter"]=route_name; rec["router_confidence"]=route_confidence
                route_counts[route_name]=route_counts.get(route_name,0)+1
            if route_weights is not None: rec["router_weights"]=route_weights
            if "output" in row: rec["gold"]=normalize_items(args.task,row["output"])
            f.write(json.dumps(rec,ensure_ascii=False)+"\n"); saved_records.append(rec)
            route_msg=f" route={route_name}({route_confidence:.3f})" if route_name is not None else ""
            print(f"[{i}/{len(rows)}]{route_msg} {row['input'][:60]} -> {json.dumps(parsed,ensure_ascii=False)}")
    info={"output":str(output),"samples":len(rows),"inference_seconds":total_seconds,
          "seconds_per_sample":total_seconds/max(len(rows),1)}
    if router_dir:
        info["router_dir"]=str(router_dir); info["router_format"]=manifest.get("format"); info["router_counts"]=route_counts
    from .tracking import log_prediction_samples
    log_prediction_samples(args,saved_records)
    print(f"\nPredictions saved to: {output}")
    print(json.dumps(info,ensure_ascii=False,indent=2))
    # free GPU memory to avoid OOM when predict is called repeatedly (e.g., NER->RE inside cdtir-predict)
    import gc
    del model
    if router is not None:
        del router
    gc.collect()
    torch.cuda.empty_cache()
    return info
