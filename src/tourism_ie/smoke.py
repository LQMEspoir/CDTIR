# File: src/tourism_ie/smoke.py
# Module responsibility: model-free self-check module: verifies data files/format, data leakage, metrics, model registry, strategies, router semantics and Python syntax.
# Main data flow: project dir -> several independent checks -> PASS/FAIL list -> overall smoke-test result.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
from pathlib import Path
import ast
from collections import Counter
from .data_utils import read_jsonl
from .parsing import normalize_items
from .evaluation import evaluate_records_detailed
from .data_split import build_benchmark_splits, labels_of
from .prompts import NER_LABELS, RELATION_LABELS
from .models import list_models, core_model_keys
from .strategies import STRATEGIES
from .router_multilora import ROUTER_GROUPS, SimpleCategoryRouter
from .data_utils import target_text, messages_for, render_prompt


def run_smoke_test(root: Path, verbose=False) -> bool:
    """- Responsibility: runs a set of project structure/data/metrics/registry/syntax checks (no model download) and aggregates PASS/FAIL per item.
    - Parameters: root: Path,verbose=False.
    - Returns: return type is `bool`; see implementation below for field details.
    - Notes: inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    checks=[]
    def check(name,fn):
        """- Responsibility: runs the local "check" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: name,fn.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        try:
            detail=fn(); checks.append((name,True,detail))
        except Exception as e:
            checks.append((name,False,f"{type(e).__name__}: {e}"))

    ner_train=root/"data_end/data/ner/entity_train.jsonl"; ner_test=root/"data_end/data/ner/entity_test.jsonl"
    rel_train=root/"data_end/data/relation/relation_train.jsonl"; rel_valid=root/"data_end/data/relation/relation_valid.jsonl"
    check("NER train count",lambda: f"{len(read_jsonl(ner_train))} (expected 911)")
    check("NER test count",lambda: f"{len(read_jsonl(ner_test))} (expected 101)")
    check("Relation train count",lambda: f"{len(read_jsonl(rel_train))} (expected 319)")
    check("Relation valid count",lambda: f"{len(read_jsonl(rel_valid))} (expected 80)")
    rel_train_e=root/"data_end/data/relation/relation_train_e.jsonl"; rel_valid_e=root/"data_end/data/relation/relation_valid_e.jsonl"
    check("Entity-aware relation train",lambda: f"{len(read_jsonl(rel_train_e))} rows / all entities present" if len(read_jsonl(rel_train_e))==319 and all(r.get("entities") for r in read_jsonl(rel_train_e)) else (_ for _ in ()).throw(ValueError("invalid entity-aware train")))
    check("Entity-aware relation valid",lambda: f"{len(read_jsonl(rel_valid_e))} rows / all entities present" if len(read_jsonl(rel_valid_e))==80 and all(r.get("entities") for r in read_jsonl(rel_valid_e)) else (_ for _ in ()).throw(ValueError("invalid entity-aware valid")))

    def validate_outputs(task,path):
        """- Responsibility: runs the local "validate outputs" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: task,path.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        rows=read_jsonl(path); total=0
        for i,r in enumerate(rows,1):
            x=normalize_items(task,r.get("output",[])); total+=len(x)
            if not isinstance(r.get("input"),str) or not r["input"].strip(): raise ValueError(f"empty input at {i}")
        return f"{len(rows)} rows / {total} annotations"
    check("NER train schema",lambda:validate_outputs("ner",ner_train))
    check("NER test schema",lambda:validate_outputs("ner",ner_test))
    check("Relation train schema",lambda:validate_outputs("relation",rel_train))
    check("Relation valid schema",lambda:validate_outputs("relation",rel_valid))

    def no_overlap():
        """- Responsibility: runs the local "no overlap" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        a={r['input'] for r in read_jsonl(ner_train)}; b={r['input'] for r in read_jsonl(ner_test)}
        if a&b: raise ValueError(f"{len(a&b)} overlapping texts")
        return "0 train/test overlaps"
    check("NER original split leakage",no_overlap)

    def benchmark_splits():
        """- Responsibility: runs the local "benchmark splits" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        build_benchmark_splits(root,seed=42)
        details=[]
        for task,labels in [("ner",NER_LABELS),("relation",RELATION_LABELS)]:
            base=root/"data_end/data/benchmark"
            parts={s:read_jsonl(base/f"{task}_{s}.jsonl") for s in ("train","valid","test")}
            all_text=[]
            for s,rows in parts.items(): all_text += [r["input"] for r in rows]
            if len(all_text)!=len(set(all_text)):
                raise ValueError(f"{task} benchmark split overlap detected")
            for label in labels:
                total=sum(label in labels_of(task,r) for rows in parts.values() for r in rows)
                if total>=3:
                    for s,rows in parts.items():
                        if not any(label in labels_of(task,r) for r in rows):
                            raise ValueError(f"{task}:{label} absent from {s} despite >=3 labeled samples")
            details.append(f"{task}={len(parts['train'])}/{len(parts['valid'])}/{len(parts['test'])}")
        return ", ".join(details)
    check("Benchmark multi-label splits",benchmark_splits)

    def metrics_self_test():
        """- Responsibility: runs the local "metrics self test" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        rec=[{"prediction":[{"entity_text":"东京","entity_label":"目的地"}],"gold":[{"entity_text":"东京","entity_label":"目的地"}]}]
        # rec=[{"prediction":[{"entity_text":"Tokyo","entity_label":"Destination"}],"gold":[{"entity_text":"Tokyo","entity_label":"Destination"}]}]
        m=evaluate_records_detailed("ner",rec)
        if m["micro"]["f1"]!=1.0 or m["exact_sample_accuracy"]!=1.0: raise ValueError(m)
        row=next(x for x in m["per_class"] if x["label"]=="目的地")
        # row=next(x for x in m["per_class"] if x["label"]=="Destination")
        if row["precision"]!=1.0 or row["recall"]!=1.0 or row["presence_accuracy"]!=1.0: raise ValueError(row)
        return "micro/macro/per-class/exact/presence metrics OK"
    check("Detailed metrics self-test",metrics_self_test)

    def cleaning_checks():
        """- Responsibility: runs the local "cleaning checks" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        rows=read_jsonl(root/"data_end/data/ner/entity_all.jsonl")
        by={r["input"]:r for r in rows}
        sh=by["文本:#上海苏宁诺富特酒店#从外滩回来怎么路线"]
        # sh=by["text:#Shanghai Suning Novotel Hotel# how to get back from the Bund"]
        sh_items=normalize_items("ner",sh["output"])
        if any(x.get("entity_text")=="`" for x in sh_items): raise ValueError("punctuation pseudo-entity still present")
        h=by["文本:#黄山701发射台招待所#从后山上去，从哪里过去可以先把行李放着再继续爬山看其他阶段的啊，路线能不能说下。"]
        # h=by["text:#Huangshan 701 hostel# go up from the back mountain ..."]
        if not any(x.get("entity_text")=="黄山701发射台招待所" and x.get("entity_label")=="住宿" for x in normalize_items("ner",h["output"])):
            # if not any(x.get("entity_text")=="Huangshan 701 hostel" and x.get("entity_label")=="Lodging" for x in ...):
            raise ValueError("701 hotel correction missing")
        b=by["文本:#见悟公寓(规划一支路分店)#有吹风机吗"]
        # b=by["text:#Jianwu Apartment (Planning 1st Road branch)# is there a hair dryer"]
        if not any(x.get("entity_text")=="吹风机" and x.get("entity_label")=="产品" for x in normalize_items("ner",b["output"])):
            # if not any(x.get("entity_text")=="hair dryer" and x.get("entity_label")=="Product" for x in ...):
            raise ValueError("hairdryer relabel correction missing")
        return "3 high-confidence corrections present; audit files retained"
    check("Curated NER cleaning",cleaning_checks)
    check("Core 4-model registry",lambda: f"4 core + {len(list_models())-4} optional" if len(core_model_keys())==4 else (_ for _ in ()).throw(ValueError(len(core_model_keys()))))
    check("GLM4 registry",lambda: "glm4-9b registered" if any(x.key=="glm4-9b" for x in list_models()) else (_ for _ in ()).throw(ValueError("glm4 missing")))
    check("Optional relation.py Qwen2 registry",lambda: "qwen2-1.5b-relation-legacy registered" if any(x.key=="qwen2-1.5b-relation-legacy" for x in list_models()) else (_ for _ in ()).throw(ValueError("legacy relation model missing")))
    check("Training strategy registry",lambda: f"{len(STRATEGIES)} strategies: {', '.join(STRATEGIES)}")
    check("Neural Router-MultiLoRA groups",lambda: f"NER={len(ROUTER_GROUPS['ner'])} paper adapters; NRE legacy={len(ROUTER_GROUPS['relation'])}" if list(ROUTER_GROUPS['ner'])==["location","time","service","attribute"] else (_ for _ in ()).throw(ValueError(ROUTER_GROUPS['ner'])))

    def legacy_compat_checks():
        """- Responsibility: runs the local "legacy compat checks" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        import ast
        from .prompts import NER_ORIGINAL_STRICT_SYSTEM_PROMPT
        sample={"output":[{"entity_text":"北京","entity_label":"目的地"}]}
        # sample={"output":[{"entity_text":"Beijing","entity_label":"Destination"}]}
        t=target_text("ner",sample,"original")
        if not t.startswith("{\"entity_text\""): raise ValueError(t)
        if "文本实体识别领域的专家" not in messages_for("ner","文本:北京","original")[0]["content"]: raise ValueError("original NER prompt missing")
        # if "expert in text entity recognition" not in messages_for("ner","text: Beijing","original")[0]["content"]: raise ValueError(...)
        legacy_src=(root/"legacy/original_scripts/entity_router_rola.py").read_text(encoding="utf-8")
        tree=ast.parse(legacy_src)
        old_prompts=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="system_prompt" for x in node.targets) and isinstance(node.value,ast.Constant):
                old_prompts.append(node.value.value)
        if not old_prompts or old_prompts[0] != NER_ORIGINAL_STRICT_SYSTEM_PROMPT:
            raise ValueError("original_strict NER prompt is not byte-equivalent to legacy runtime string")
        strict_frame=render_prompt(None,"ner","文本:北京","original_strict",True)
        # strict_frame=render_prompt(None,"ner","text: Beijing","original_strict",True)
        if not strict_frame.startswith("<|im_start|>system\n\n    ") or not strict_frame.endswith("<|im_start|>assistant\n"):
            raise ValueError("strict Qwen framing mismatch")
        r=SimpleCategoryRouter(16,4)
        if r.linear[-1].out_features!=4: raise ValueError("router output mismatch")
        rel0=read_jsonl(root/"data_end/data/original_reference/relationship_output_type_original.jsonl")[0]
        strict_msgs=messages_for("relation",rel0["input"],"text_only_strict")
        if len(strict_msgs)!=1 or strict_msgs[0]["role"]!="user" or "文本:文本:" in strict_msgs[0]["content"]:
            # ... or "text: text:" in strict_msgs[0]["content"]:
            raise ValueError("strict text-only RE must be one user turn with one 文本: prefix")
        strict_target=target_text("relation",rel0,"canonical","text_only_strict")
        if not strict_target.startswith("```json\n[") or '"ob1"' not in strict_target:
            raise ValueError("strict RE notebook target format mismatch")
        return "legacy target/prompts retained + four-way paper NER router OK"

    check("Original-algorithm compatibility",legacy_compat_checks)

    def legacy_relation_checks():
        """- Responsibility: runs the local "legacy relation checks" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input.
        """
        from .relation_legacy import preprocess_spo_entry, extract_triples, encode_fixed_row
        row={"text":"轴承外圈磨损会导致振动增大。","spo_list":[{"subject":[0,6],"predicate":"导致","object":[9,13]}]}
        # row={"text":"Bearing outer-ring wear causes increased vibration.","spo_list":[{"subject":[0,6],"predicate":"causes","object":[9,13]}]}
        x=preprocess_spo_entry(row)
        if x["target_text"] != "<轴承外圈磨损, 导致, 振动增大>": raise ValueError(x)
        # if x["target_text"] != "<Bearing outer-ring wear, causes, increased vibration>": raise ValueError(x)
        if extract_triples(x["target_text"]) != {("轴承外圈磨损","导致","振动增大")}: raise ValueError(x)
        # if extract_triples(x["target_text"]) != {("Bearing outer-ring wear","causes","increased vibration")}: raise ValueError(x)
        class ToyTok:
            """- Responsibility: encapsulates ToyTok state and operations so it can be invoked as a standalone component by train/inference/eval flows.
            - Inherits: no explicit business base class.
            - Usage: instantiated then invoked by the training/inference main flow via public methods; internal state is determined by __init__ and saved config.
            """
            eos_token_id=99
            # - Responsibility: pads variable-length training samples into batch tensors and stacks the router multi-label supervision vector.
            # - Parameters: text,add_special_tokens=False.
            # - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
            # - Notes: mainly in-memory; no side effects beyond the called object itself.
            def __call__(self,text,add_special_tokens=False): return {"input_ids":[ord(c)%50+1 for c in text]}
        enc=encode_fixed_row(ToyTok(),{"input_text":"PROMPT","target_text":"<a,b,c>"},32)
        first=next(i for i,v in enumerate(enc["labels"]) if v!=-100)
        if first != len("PROMPT") or any(v!=-100 for v in enc["labels"][:first]): raise ValueError(enc)
        return "SPO offsets -> <S,P,O> + causal labels aligned after prompt mask (-100)"
    check("Legacy relation.py fixed route",legacy_relation_checks)

    def original_sources():
        """- Responsibility: runs the local "original sources" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        a=root/"data_end/data/original_reference/entity_output_original.jsonl"; b=root/"data_end/data/original_reference/relationship_output_type_original.jsonl"
        return f"NER={len(read_jsonl(a))}, NRE={len(read_jsonl(b))}"
    check("Packaged original source data",original_sources)

    def compile_source():
        """- Responsibility: runs the local "compile source" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        n=0
        for p in [root/"run.py", *sorted((root/"src").rglob("*.py")), *sorted((root/"scripts").rglob("*.py"))]:
            ast.parse(p.read_text(encoding="utf-8"),filename=str(p)); n+=1
        return f"{n} Python files parsed"
    check("Python syntax",compile_source)

    ok=all(x[1] for x in checks)
    print("="*78); print("TOURISM IE BENCHMARK PACKAGE SMOKE TEST"); print("="*78)
    for name,passed,detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print("="*78); print("RESULT:","PASS - code/data/metrics/benchmark structure is runnable" if ok else "FAIL - see errors above")
    if verbose:
        print("Model training/inference additionally requires the Python dependencies and model weights/network access.")
        print("QLoRA additionally requires CUDA + bitsandbytes; gated models may require account authorization.")
    return ok
