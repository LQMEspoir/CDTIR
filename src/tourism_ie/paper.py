"""Paper-reproduction presets, data audit, pipeline, reports and manifest."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from argparse import Namespace
from collections import Counter
from pathlib import Path

from .prompts import NER_LABELS, RELATION_LABELS

VARIANTS=("CDTIR","CDTIR_NMoRE","CDTIR_F","CDTIR_EH","CDTIR_TO")
PAPER_DEFAULTS={
    "model":"qwen2.5-7b","lora_r":8,"lora_alpha":32,"lora_dropout":0.1,
    "batch_size":2,"eval_batch_size":2,"grad_accum":2,"learning_rate":3e-5,
    "weight_decay":0.01,"warmup_ratio":0.01,"lr_scheduler_type":"cosine",
    "max_grad_norm":1.0,"epochs":4.0,"router_loss_weight":0.5,"seed":2026,
    "logging_steps":10,"eval_strategy":"steps","eval_steps":20,
    "save_strategy":"steps","save_steps":20,"save_total_limit":2,
}
PRESETS={
    "CDTIR":{"ner_strategy":"router_multilora","ner_prompt":"paper_ner","re_strategy":"balanced_lora","re_prompt":"paper_re_e"},
    "CDTIR_NMoRE":{"ner_strategy":"lora","ner_prompt":"paper_ner","re_strategy":"reuse:CDTIR","re_prompt":"paper_re_e"},
    "CDTIR_F":{"ner_strategy":"zero_shot","ner_prompt":"paper_ner","re_strategy":"zero_shot","re_prompt":"paper_re_e"},
    "CDTIR_EH":{"ner_strategy":"reuse:CDTIR","ner_prompt":"paper_ner","re_strategy":"balanced_lora","re_prompt":"paper_re_eh"},
    "CDTIR_TO":{"ner_strategy":"reuse:CDTIR","ner_prompt":"paper_ner","re_strategy":"balanced_lora","re_prompt":"paper_re_to"},
}
RELATION_TYPE_PAIRS={
    "体验内容":({"目的地","餐饮","住宿"},{"产品"}),
    # "体验内容":({"Destination","Dining","Lodging"},{"Product"})
    "预算限制":({"目的地","餐饮","住宿","产品","交通"},{"预算"}),
    # "预算限制":({"Destination","Dining","Lodging","Product","Transport"},{"Budget"})
    "人均预算":({"预算"},{"人数"}),
    # "人均预算":({"Budget"},{"People"})
    "停留时长":({"目的地","餐饮","住宿","产品"},{"时长"}),
    # "停留时长":({"Destination","Dining","Lodging","Product"},{"Duration"})
    "出行时间":({"出发地","返程地","目的地","餐饮","住宿","产品"},{"时间"}),
    # "出行时间":({"Origin","Return","Destination","Dining","Lodging","Product"},{"Time"})
    "游玩顺序":({"出发地","目的地"},{"目的地","返程地"}),
    # "游玩顺序":({"Origin","Destination"},{"Destination","Return"})
}


def _read(path:Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _paper_root(root: Path) -> Path:
    """Resolve paper artifacts relative to the runtime workspace.

    ``CDTIR_PAPER_OUTPUT_ROOT`` is optional and may be absolute or relative to
    the current workspace.  With no override, all artifacts are written to
    ``<current-workspace>/outputs/paper``.
    """
    configured=os.environ.get("CDTIR_PAPER_OUTPUT_ROOT")
    if not configured:
        return root/"outputs"/"paper"
    path=Path(configured).expanduser()
    return path if path.is_absolute() else root/path


def _write(path:Path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")


def _sha(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def validate_paper_data(root:Path, data_dir="data_end/data_merged", write_reports=True):
    base=root/data_dir; members=_read(base/"split_membership.jsonl")
    if not members:
        raise AssertionError("paper data is empty (no split_membership rows)")
    unique={x["text_id"] for x in members}
    if len(members)!=len(unique):
        raise AssertionError("paper data must have unique text_id values")
    split_counts=Counter(x["split"] for x in members)
    if set(split_counts)-{"train","valid","test"}:
        raise AssertionError(f"unexpected split labels: {dict(split_counts)}")
    membership={x["text_id"]:x["split"] for x in members}
    distributions=[]; seen={}
    for task,labels in (("ner",NER_LABELS),("relation",RELATION_LABELS)):
        for split in ("train","valid","test"):
            suffix="entity" if task=="ner" else "relation"
            rows=_read(base/task/f"{suffix}_{split}.jsonl")
            ids={r["text_id"] for r in rows}; seen[(task,split)]=ids
            if any(membership[i]!=split for i in ids): raise AssertionError(f"{task}/{split} membership mismatch")
            counts=Counter()
            for r in rows:
                for x in r.get("output",[]): counts[x["entity_label"] if task=="ner" else x.get("predicate",x.get("rel"))]+=1
            for label in labels: distributions.append({"task":task,"split":split,"label":label,"count":counts[label]})
    for split in ("train","valid","test"):
        if seen[("ner",split)] != seen[("relation",split)]: raise AssertionError(f"NER/RE mismatch: {split}")
    if any(seen[("ner",a)] & seen[("ner",b)] for a,b in (("train","valid"),("train","test"),("valid","test"))):
        raise AssertionError("cross-split leakage")
    invalid_relations=[]
    for row in _read(base/"relation"/"relation_all_e.jsonl"):
        types={}
        for e in row.get("entities",[]): types.setdefault(e["entity_text"],set()).add(e["entity_label"])
        for rel in row.get("output",[]):
            s,p,o=rel.get("subject"),rel.get("predicate"),rel.get("object")
            pair=RELATION_TYPE_PAIRS.get(p)
            valid=bool(pair and s in types and o in types and (types[s]&pair[0]) and (types[o]&pair[1]))
            if p=="游玩顺序" and valid:  # Play Order
                valid=bool(("出发地" in types[s] and "目的地" in types[o]) or  # Origin -> Destination
                           ("目的地" in types[s] and ({"目的地","返程地"}&types[o])))  # Destination -> Destination/Return
            if not valid: invalid_relations.append({"text_id":row["text_id"],"relation":rel})
    if invalid_relations: raise AssertionError(f"invalid relation triples: {invalid_relations[:3]}")
    if write_reports:
        with (base/"label_distribution.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["task","split","label","count"]); w.writeheader(); w.writerows(distributions)
        status=json.loads((base/"generation_report.json").read_text(encoding="utf-8"))
        review_done=status.get("human_review_status")=="completed"
        report={"status":"valid","unique_text_ids":len(unique),"split_counts":dict(split_counts),
                "ner_re_membership_identical":True,"cross_split_leakage":0,"invalid_relation_triples":0,
                "relation_annotation":status,
                "truth_status":"human_verified" if review_done else "pending_human_review",
                "truth_caveat":None if review_done else "Ollama-generated rows remain pending human semantic review; they are not labelled as manual ground truth."}
        (base/"data_build_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        progress=base/"relation_generation"/"progress.jsonl"
        if progress.exists():
            audit=[]
            prior={}
            prior_path=base/"relation_annotation_audit.jsonl"
            if prior_path.exists(): prior={x["text_id"]:x for x in _read(prior_path)}
            old_by_text={r["input"]:r.get("output",[]) for r in _read(root/"data_end/data"/"relation"/"relation_all.jsonl")}
            for r in _read(progress):
                item={"text_id":r["text_id"],"source":r.get("source"),
                    "old_value":old_by_text.get(r["input"]),"new_value":r.get("output",[]),
                    "reasons":[x.get("reason") for x in r.get("rejected",[])],
                    "confidence":None,"review_status":"pending" if r.get("source") in {"ollama_generated_reviewed","existing_gold_repaired"} else "rule_or_existing"}
                old=prior.get(r["text_id"],{})
                if old.get("review_status")=="human_verified" and old.get("new_value")==item["new_value"]:
                    item.update({k:old[k] for k in ("confidence","review_status","reviewer","reviewed_at","review_reason") if k in old})
                audit.append(item)
            _write(base/"relation_annotation_audit.jsonl",audit)
        correction_source=root/"data_end/data"/"cleaning"/"applied_corrections.json"
        if correction_source.exists():
            shutil.copy2(correction_source,base/"correction_audit.json")
        build_manifest(root,data_dir)
    return {"unique_text_ids":len(unique),"split_counts":dict(split_counts),"valid":True}


def build_manifest(root:Path, data_dir="data_end/data_merged"):
    base=root/data_dir; files=[]
    split_label=""
    sm=base/"split_membership.jsonl"
    if sm.exists():
        c=Counter(json.loads(x).get("split") for x in sm.read_text(encoding="utf-8").splitlines() if x.strip())
        split_label="/".join(str(c.get(k,0)) for k in ("train","valid","test"))
    excluded={".git",".idea","__pycache__","outputs"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name!="MANIFEST.json" and not any(part in excluded for part in p.parts) and p.suffix!=".pyc":
            lines=sum(1 for _ in p.open("rb")) if p.suffix in {".jsonl",".csv"} else None
            files.append({"path":p.relative_to(root).as_posix(),"sha256":_sha(p),"lines":lines})
    payload={"format":"cdtir-paper-manifest-v1","seed":2026,"split":split_label,
             "prompt_version":"table-4-6-4-7-v2","presets":PRESETS,"paper_defaults":PAPER_DEFAULTS,"files":files}
    (root/"MANIFEST.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return payload


def preset_snapshot(variant):
    if variant not in PRESETS: raise ValueError(f"variant must be one of {VARIANTS}")
    return {"variant":variant,**PRESETS[variant],**PAPER_DEFAULTS,"ner_max_length":512,"re_max_length":1200,
            "lora_target_modules":["q_proj","k_proj","v_proj","o_proj"]}


def paper_train(root:Path,args):
    from .training import train
    spec=preset_snapshot(args.variant); out=_paper_root(root)/args.variant
    (out).mkdir(parents=True,exist_ok=True)
    (out/"preset.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding="utf-8")
    results={}
    tasks=("ner","relation") if args.task=="both" else (args.task,)
    for task in tasks:
        strategy=spec[f"{'ner' if task=='ner' else 're'}_strategy"]
        if strategy=="zero_shot" or strategy.startswith("reuse:"):
            results[task]={"status":strategy}; continue
        values={**PAPER_DEFAULTS,"task":task,"strategy":strategy,"prompt_profile":spec[f"{'ner' if task=='ner' else 're'}_prompt"],
                "target_format":"original","max_length":512 if task=="ner" else 1200,"split":"benchmark",
                "data_dir":str(root/"data_end/data_merged"),"output_dir":str(out/task),"max_steps":args.max_steps,
                "resume_from_checkpoint":args.resume_from_checkpoint,"empty_relation_weight":getattr(args,"empty_relation_weight",1.0),"gradient_checkpointing":True,"limit":args.limit,
                "focal_alpha":0.25,"focal_gamma":2.0,"balance_max_multiplier":5,
                "router_forward_top_k":0,"router_checkpoint_mixture":True,"router_hidden_mode":"frozen_base",
                "router_adapter_epochs":1.0,"router_threshold":0.45,"swanlab":False,"swanlab_project":None,"swanlab_experiment":None}
        # A five-step smoke must still exercise evaluation, token accuracy and a
        # resumable checkpoint; the full paper preset remains every 20 steps.
        if 0 < args.max_steps <= 5:
            values.update({"logging_steps":1,"eval_steps":args.max_steps,"save_steps":args.max_steps})
        if args.model: values["model"]=args.model
        results[task]=train(root,Namespace(**values))
    return results


def _predict_args(**kw):
    defaults=dict(model="qwen2.5-7b",adapter=None,router_dir=None,router_threshold=None,
        router_generation_mode="top1",router_generation_top_k=0,load_in_4bit=False,
        split="benchmark",eval_split="test",text=None,entities_json=None,input_file=None,
        dataset_file=None,output=None,max_new_tokens=256,limit=0,prompt_profile="paper_ner",
        data_dir=None,swanlab=False,swanlab_project=None,swanlab_experiment=None)
    defaults.update(kw); return Namespace(**defaults)


def build_pipeline_inputs(ner_predictions, relation_gold_rows):
    """Join only by text_id; Pred entities never read the Gold entities field."""
    rel_gold={r["text_id"]:r for r in relation_gold_rows}; pred_rows=[]; gold_rows=[]
    for r in ner_predictions:
        g=rel_gold[r["text_id"]]
        pred_rows.append({"text_id":r["text_id"],"input":g["input"],
                          "entities":r.get("prediction",[]),"output":g["output"]})
        gold_rows.append(g)
    return pred_rows,gold_rows


def cdtir_predict(root:Path,args):
    """Run NER first, then RE with Pred entities and independently with Gold entities."""
    from .inference import predict
    out=Path(args.output_dir) if args.output_dir else _paper_root(root)/args.variant/"pipeline"
    out.mkdir(parents=True,exist_ok=True); test=root/"data_end/data_merged"/"benchmark"/"ner_test.jsonl"
    ner_path=out/"ner_predictions.jsonl"
    predict(root,_predict_args(task="ner",model=args.model,adapter=args.ner_adapter,router_dir=args.ner_router,
        router_generation_mode="weighted",dataset_file=str(test),output=str(ner_path),
        max_new_tokens=args.max_new_tokens,prompt_profile=getattr(args,"ner_prompt_profile","paper_ner")))
    ner=_read(ner_path)
    pred_rows,gold_rows=build_pipeline_inputs(ner,_read(root/"data_end/data_merged"/"benchmark"/"relation_test_e.jsonl"))
    pred_input=out/"re_e_pred_input.jsonl"; gold_input=out/"re_e_gold_input.jsonl"
    _write(pred_input,pred_rows); _write(gold_input,gold_rows)
    common=dict(task="relation",model=args.model,adapter=args.re_adapter,
                max_new_tokens=args.max_new_tokens,prompt_profile=getattr(args,"re_prompt_profile","paper_re_e"))
    pred_result=predict(root,_predict_args(**common,dataset_file=str(pred_input),output=str(out/"re_e_pred_predictions.jsonl")))
    gold_result=predict(root,_predict_args(**common,dataset_file=str(gold_input),output=str(out/"re_e_gold_predictions.jsonl")))
    return {"ner_predictions":str(ner_path),"re_e_pred":pred_result,"re_e_gold":gold_result}


def _paper_metric_row(name, task, path, token_accuracy=""):
    from .evaluation.metrics import evaluate_records_detailed
    m=evaluate_records_detailed(task,_read(path))
    return {"system":name,"task":task,
            "micro_p":m["micro"]["precision"],"micro_r":m["micro"]["recall"],
            "micro_f1":m["micro"]["f1"],"macro_f1":m["macro"]["f1"],
            "accuracy":m["accuracy"],"token_accuracy":token_accuracy}


def paper_report(root:Path,args):
    """Write Figure 4, Table 9 (backbones) and Table 10 (ablations)."""
    from .evaluation.metrics import evaluate_records_detailed
    out=Path(args.output_dir) if args.output_dir else _paper_root(root)/"reports"; out.mkdir(parents=True,exist_ok=True)
    # Remove report names from the previous numbering scheme in this exact
    # report directory.  They are stale aliases, not user data, and keeping
    # them beside the new files makes it easy to cite the wrong table.
    for stale in ("figure_4_7.png","figure_4_7.pdf","figure_4_7.csv",
                  "table_4_11.csv","table_4_12.csv"):
        old=out/stale
        if old.exists(): old.unlink()
    # Table 9: only Qwen/GLM/LLaMA backbone experiments.
    rows=[]; backbone_names={"qwen2.5-7b":"Qwen2.5-7B","glm4-9b":"GLM-4-9B","llama3.1-8b":"LLaMA3.1-8B"}
    for task in ("ner","relation"):
        summary=root/"outputs"/"backbone_compare"/task/"summary.json"
        if not summary.exists():
            continue
        expected_strategy="router_multilora" if task=="ner" else "lora"
        for item in json.loads(summary.read_text(encoding="utf-8")):
            key=item.get("model_key")
            if key not in backbone_names or item.get("strategy") != expected_strategy or item.get("status") != "ok":
                continue
            rows.append({"backbone":backbone_names[key],"task":task,
                         "micro_p":item.get("micro_precision",""),"micro_r":item.get("micro_recall",""),
                         "micro_f1":item.get("micro_f1",""),"macro_f1":item.get("macro_f1",""),
                         "accuracy":item.get("accuracy",""),
                         "token_accuracy":item.get("token_accuracy","")})
    with (out/"table_9.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["backbone","task","micro_p","micro_r","micro_f1","macro_f1","accuracy","token_accuracy"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    # Table 10: CDTIR and exactly four ablation variants.  Main CDTIR RE is
    # always the real NER->RE Pred pipeline, never the Gold-entity upper bound.
    table12=[]
    for variant in VARIANTS:
        for task in ("ner","relation"):
            if task=="relation" and variant in {"CDTIR","CDTIR_NMoRE","CDTIR_F"}:
                p=_paper_root(root)/variant/"pipeline"/"re_e_pred_predictions.jsonl"
            else:
                p=_paper_root(root)/variant/"predictions"/f"{task}.jsonl"
            if p.exists():
                hist=_paper_root(root)/variant/task/"training_report.json"
                if not hist.exists():
                    token_sources={
                        ("CDTIR_NMoRE","relation"):("CDTIR","relation"),
                        ("CDTIR_EH","ner"):("CDTIR","ner"),
                        ("CDTIR_TO","ner"):("CDTIR","ner"),
                    }
                    source=token_sources.get((variant,task))
                    if source:
                        hist=_paper_root(root)/source[0]/source[1]/"training_report.json"
                token_acc=""
                if hist.exists(): token_acc=json.loads(hist.read_text(encoding="utf-8")).get("best_eval_token_acc","")
                table12.append(_paper_metric_row(variant,task,p,token_acc))
    with (out/"table_10.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields12=["system","task","micro_p","micro_r","micro_f1","macro_f1","accuracy","token_accuracy"]
        w=csv.DictWriter(f,fieldnames=fields12); w.writeheader(); w.writerows(table12)

    # Gold entities are an explicitly separate upper-bound artifact.
    upper=[]
    paths={
        "CDTIR RE-E-Pred":_paper_root(root)/"CDTIR"/"pipeline"/"re_e_pred_predictions.jsonl",
        "RE-E-Gold upper bound":_paper_root(root)/"CDTIR"/"pipeline"/"re_e_gold_predictions.jsonl",
    }
    for name,p in paths.items():
        if p.exists():
            upper.append(_paper_metric_row(name,"relation",p))
    with (out/"re_e_gold_upper_bound.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["system","task","micro_p","micro_r","micro_f1","macro_f1","accuracy","token_accuracy"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(upper)
    _plot_figure_4(root,out)
    return {"figure_4":str(out/"figure_4.png"),"table_9":str(out/"table_9.csv"),"table_10":str(out/"table_10.csv"),
            "re_e_gold_upper_bound":str(out/"re_e_gold_upper_bound.csv"),"output_dir":str(out)}


def _plot_figure_4(root:Path,out:Path):
    histories={}
    for task, dname in (("ner","ner"),("relation","re")):
        tr=root/"outputs"/"backbone_qwen2.5-7b"/dname/"training_report.json"
        if tr.exists():
            histories[task]=json.loads(tr.read_text(encoding="utf-8")).get("eval_history",[])
    fields=["task","epoch","step","eval_loss","eval_token_acc"]
    with (out/"figure_4.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for task,h in histories.items():
            for x in h: w.writerow({"task":task,**x})
    if len(histories)!=2: return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE="#0072B2"; ORANGE="#E69F00"
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for task,name,color in (("ner","NER",BLUE),("relation","RE",ORANGE)):
        h=histories[task]; epochs=[x["epoch"] for x in h]
        ax[0].plot(epochs,[x["eval_loss"] for x in h],color=color,linewidth=1.8,label=f"{name}_Loss")
        ax[1].plot(epochs,[x["eval_token_acc"] for x in h],color=color,linewidth=1.8,label=f"{name}_Token_Acc")
    ax[0].set(xlabel="Epoch",ylabel="Loss"); ax[1].set(xlabel="Epoch",ylabel="Token Accuracy")
    for a in ax:
        a.legend(frameon=False); a.grid(alpha=.25)
        a.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out/"figure_4.png",dpi=300); fig.savefig(out/"figure_4.pdf"); plt.close(fig)
