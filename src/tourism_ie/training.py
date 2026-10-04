# File: src/tourism_ie/training.py
# Module responsibility: unified training entry: sample encoding, strategy resolution, LoRA/QLoRA config, Focal Trainer, Trainer args and final adapter save.
# Main data flow: task/model/strategy + data split -> tokenizer/model -> dataset encoding -> Trainer/PEFT training -> final adapter + manifest/stats.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
from pathlib import Path
import json, time
import torch
from .data_utils import read_jsonl, task_paths, target_text, messages_for, render_prompt, balance_rows, is_entity_aware_profile, require_entity_aware_rows
from .modeling import load_tokenizer, load_base_model, find_lora_targets, count_parameters
from .models import resolve_model
from .strategies import get_strategy


def swift_token_accuracy(logits_or_ids, labels) -> float:
    """Swift-compatible causal-LM token accuracy (shift, then labels != -100)."""
    import numpy as np
    pred=np.asarray(logits_or_ids); gold=np.asarray(labels)
    if pred.ndim == gold.ndim + 1: pred=pred.argmax(axis=-1)
    pred=pred[..., :-1]; gold=gold[..., 1:]; mask=gold != -100
    return float((pred[mask] == gold[mask]).mean()) if mask.any() else 0.0


def _compute_token_metrics(eval_pred):
    return {"token_acc":swift_token_accuracy(eval_pred.predictions,eval_pred.label_ids)}


def _argmax_logits(logits, labels):
    if isinstance(logits,tuple): logits=logits[0]
    return logits.argmax(dim=-1)


def _resume_value(value, output_dir=None, max_steps=-1):
    if not value: return None
    if str(value).lower()!="latest": return str(value)
    if output_dir:
        root=Path(output_dir)
        checkpoints=[]
        for p in root.glob("checkpoint-*"):
            try: checkpoints.append((int(p.name.split("-")[1]),p))
            except (IndexError,ValueError): continue
        if checkpoints:
            step,path=max(checkpoints)
            if int(max_steps)>0 and step>=int(max_steps):
                raise ValueError(f"Latest checkpoint step {step} already reached --max-steps {max_steps}: {path}")
            return str(path)
    return True


def _encode_dataset(tokenizer, task, rows, max_length, prompt_profile="canonical", target_format="canonical", empty_relation_weight=1.0):
    """- Responsibility: encodes task samples via the chat template into causal-LM input_ids/attention_mask/labels, masking the prompt part.
    - Parameters: tokenizer,task,rows,max_length,prompt_profile='canonical',target_format='canonical'.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: inference/tensor paths must keep device and dtype consistent.
    """
    from datasets import Dataset
    def encode(row):
        """- Responsibility: runs the local "encode" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: row.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: inference/tensor paths must keep device and dtype consistent.
        """
        # E-branch: extract entities from row if available
        entities = row.get("entities", []) if task == "relation" and is_entity_aware_profile(prompt_profile) else None
        prompt=render_prompt(tokenizer,task,row["input"],prompt_profile,add_generation_prompt=True,entities=entities)
        answer=target_text(task,row,target_format,prompt_profile)
        p=tokenizer(prompt,add_special_tokens=False)["input_ids"]
        a=tokenizer(answer,add_special_tokens=False)["input_ids"]
        if task=="ner" and target_format=="original":
            # Legacy entity_QW2.5*.py appended tokenizer.pad_token_id to the response.
            a=a+[tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id]
        elif tokenizer.eos_token_id is not None:
            a=a+[tokenizer.eos_token_id]
        ids=(p+a)[:max_length]
        labels=([-100]*len(p)+a)[:max_length]
        weight=empty_relation_weight if (task=="relation" and not row.get("output",[])) else 1.0
        return {"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels,"weight":weight}
    return Dataset.from_list(rows).map(encode, remove_columns=list(rows[0].keys()))


def _make_focal_trainer(base_cls, gamma: float, alpha: float):
    """- Responsibility: dynamically builds a Trainer subclass overriding compute_loss to apply Focal Loss on token-level cross entropy.
    - Parameters: base_cls,gamma: float,alpha: float.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    import torch.nn.functional as F
    class FocalTrainer(base_cls):
        """- Responsibility: encapsulates FocalTrainer state and operations so it can be invoked as a standalone component by train/inference/eval flows.
        - Inherits: base_cls.
        - Usage: instantiated then invoked by the training/inference main flow via public methods; internal state is determined by __init__ and saved config.
        """
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """- Responsibility: runs the local "compute loss" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
            - Parameters: model,inputs,return_outputs=False,**kwargs.
            - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
            - Notes: mainly in-memory; no side effects beyond the called object itself.
            """
            labels=inputs.get("labels")
            outputs=model(**inputs)
            logits=outputs.logits
            shift_logits=logits[..., :-1, :].contiguous()
            shift_labels=labels[..., 1:].contiguous()
            ce=F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="none")
            mask=shift_labels.view(-1).ne(-100)
            ce=ce[mask]
            if ce.numel()==0:
                loss=outputs.loss
            else:
                pt=torch.exp(-ce)
                # Matches entity_QW2.5_fl.py: alpha * (1-pt)^gamma * CE.
                loss=(alpha*((1-pt)**gamma)*ce).mean()
            return (loss, outputs) if return_outputs else loss
    return FocalTrainer


def _weighted_collator(tokenizer):
    from transformers import DataCollatorForSeq2Seq
    class WeightedCollator(DataCollatorForSeq2Seq):
        def __call__(self, features, return_tensors=None):
            weights=[f.pop("weight",1.0) for f in features]
            batch=super().__call__(features,return_tensors=return_tensors)
            batch["weight"]=torch.tensor(weights,dtype=torch.float32)
            return batch
    return WeightedCollator(tokenizer=tokenizer,padding=True,label_pad_token_id=-100)


def _make_weighted_trainer(base_cls):
    import torch.nn.functional as F
    class WeightedTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights=inputs.pop("weight",None)
            labels=inputs.get("labels")
            outputs=model(**inputs)
            logits=outputs.logits
            shift_logits=logits[...,:-1,:].contiguous()
            shift_labels=labels[...,1:].contiguous()
            ce=F.cross_entropy(shift_logits.view(-1,shift_logits.size(-1)),shift_labels.view(-1),ignore_index=-100,reduction="none")
            ce=ce.view(shift_labels.size())
            mask=shift_labels.ne(-100).float()
            per_sample=(ce*mask).sum(dim=-1)/mask.sum(dim=-1).clamp(min=1.0)
            if weights is not None:
                weights=weights.to(per_sample.device)
                loss=(per_sample*weights).sum()/weights.sum().clamp(min=1.0)
            else:
                loss=per_sample.mean()
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def _effective_lora_hparams(args):
    """- Responsibility: resolves final LoRA rank/alpha/dropout and target_modules from model size / user input.
    - Parameters: args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    """Keep the generic benchmark defaults while allowing original-project reproduction.

    The original NER scripts used r=8, alpha=32, dropout=0.1. The tourism NRE
    notebook used r=8, alpha=16, dropout=0.0. Explicit CLI values always win.
    """
    r=args.lora_r if args.lora_r is not None else 8
    if args.lora_alpha is not None:
        alpha=args.lora_alpha
    else:
        alpha=32 if args.task=="ner" else 16
    if args.lora_dropout is not None:
        dropout=args.lora_dropout
    else:
        dropout=0.1 if args.task=="ner" else 0.0
    return int(r),int(alpha),float(dropout)


def train(root: Path, args):
    """- Responsibility: unified training entry point: resolves data and strategy, loads the model, configures PEFT/Trainer, trains and saves the final model + stats.
    - Parameters: root: Path,args.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Key process: resolve data and strategy; load tokenizer/base model; inject LoRA/quantization/sampling or focal loss per strategy; then train via Trainer and save adapter + metadata.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;involves training/backprop; memory/randomness affect resource usage and reproducibility;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    strategy=get_strategy(args.strategy)
    if not strategy.trainable:
        raise SystemExit("zero_shot does not train. Use `predict` or `benchmark --strategy zero_shot`.")
    if args.strategy == "relation_legacy_fixed":
        if args.task != "relation":
            raise SystemExit("relation_legacy_fixed is only valid with --task relation")
        from .relation_legacy import train_relation_legacy_fixed
        return train_relation_legacy_fixed(root, args)
    if args.strategy == "router_multilora":
        from .router_multilora import train_router_multilora
        return train_router_multilora(root, args)
    if args.strategy == "grouped_router_multilora":
        from .router_grouped import train_router_multilora
        return train_router_multilora(root, args)
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, set_seed
    except ImportError as e:
        raise SystemExit("Missing training dependencies. Run: pip install -r requirements.txt") from e

    set_seed(args.seed)
    prompt_profile=getattr(args,"prompt_profile","canonical")
    train_path, eval_path=task_paths(root,args.task,args.split,prompt_profile,getattr(args,"data_dir",None))
    train_rows=read_jsonl(train_path); eval_rows=read_jsonl(eval_path)
    require_entity_aware_rows(train_rows,prompt_profile,"training data")
    require_entity_aware_rows(eval_rows,prompt_profile,"evaluation data")
    if args.limit:
        train_rows=train_rows[:args.limit]
        eval_rows=eval_rows[:max(1,min(args.limit//5 or 1,len(eval_rows)))]
    original_train_count=len(train_rows)
    if strategy.balanced:
        train_rows=balance_rows(args.task,train_rows,args.balance_max_multiplier)

    spec=resolve_model(args.model)
    tokenizer=load_tokenizer(args.model)
    model=load_base_model(args.model,training=True,quantized=strategy.quantized)
    if strategy.quantized:
        model=prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing and not strategy.quantized:
        model.gradient_checkpointing_enable()
        if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()

    lora_r,lora_alpha,lora_dropout=_effective_lora_hparams(args)
    if strategy.peft:
        targets=find_lora_targets(model,spec.target_modules)
        lora=LoraConfig(task_type=TaskType.CAUSAL_LM,inference_mode=False,r=lora_r,
                        lora_alpha=lora_alpha,lora_dropout=lora_dropout,target_modules=targets)
        model=get_peft_model(model,lora)
        if hasattr(model,"print_trainable_parameters"): model.print_trainable_parameters()

    params=count_parameters(model)
    target_format=getattr(args,"target_format","canonical")
    empty_w=getattr(args,"empty_relation_weight",1.0)
    train_ds=_encode_dataset(tokenizer,args.task,train_rows,args.max_length,prompt_profile,target_format,empty_relation_weight=empty_w)
    eval_ds=_encode_dataset(tokenizer,args.task,eval_rows,args.max_length,prompt_profile,target_format,empty_relation_weight=empty_w)

    out=Path(args.output_dir) if args.output_dir else root/"outputs"/f"{args.task}_{spec.key}_{args.strategy}"
    out.mkdir(parents=True,exist_ok=True)
    use_cuda=torch.cuda.is_available(); bf16=use_cuda and torch.cuda.is_bf16_supported(); fp16=use_cuda and not bf16
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    targs=TrainingArguments(
        output_dir=str(out),num_train_epochs=args.epochs,
        max_steps=getattr(args,"max_steps",-1),
        per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,save_strategy=args.save_strategy,save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,eval_steps=args.eval_steps,lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,save_total_limit=args.save_total_limit,report_to="none",remove_unused_columns=False,
        load_best_model_at_end=(args.save_strategy==args.eval_strategy and args.save_strategy!="no"),
        metric_for_best_model="eval_loss",greater_is_better=False,
        gradient_checkpointing=args.gradient_checkpointing,bf16=bf16,fp16=fp16,
        dataloader_pin_memory=use_cuda,seed=args.seed,data_seed=args.seed,
    )
    TrainerCls=_make_focal_trainer(Trainer,args.focal_gamma,args.focal_alpha) if strategy.focal else Trainer
    collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100)
    if args.task=="relation" and empty_w < 1.0:
        TrainerCls=_make_weighted_trainer(TrainerCls)
        collator=_weighted_collator(tokenizer)
    from .tracking import build_swanlab_callbacks, finish_swanlab, EvalLossCurveCallback
    loss_cb=EvalLossCurveCallback()
    callbacks=[loss_cb]+build_swanlab_callbacks(args,{"task":args.task,"model_key":spec.key,"model_name":spec.model_name,"strategy":args.strategy,
                                            "prompt_profile":prompt_profile,"target_format":target_format,"lora_r":lora_r,
                                            "lora_alpha":lora_alpha,"lora_dropout":lora_dropout})
    trainer=TrainerCls(model=model,args=targs,train_dataset=train_ds,eval_dataset=eval_ds,
                       data_collator=collator,
                       compute_metrics=_compute_token_metrics,preprocess_logits_for_metrics=_argmax_logits,
                       callbacks=callbacks)
    start=time.perf_counter()
    try:
        train_result=trainer.train(resume_from_checkpoint=_resume_value(
            getattr(args,"resume_from_checkpoint",None),out,getattr(args,"max_steps",-1)))
    finally:
        finish_swanlab(getattr(args,"swanlab",False))
    elapsed=time.perf_counter()-start
    if strategy.peft:
        final=out/"final_adapter"
    else:
        final=out/"final_model"
    trainer.model.save_pretrained(final); tokenizer.save_pretrained(final)
    peak_gb=(torch.cuda.max_memory_allocated()/1024**3) if use_cuda else 0.0
    report={
        "task":args.task,"model_key":spec.key,"model_name":spec.model_name,"strategy":args.strategy,
        "split":args.split,"seed":args.seed,"train_rows_original":original_train_count,"train_rows_effective":len(train_rows),
        "eval_rows":len(eval_rows),"epochs":args.epochs,"training_seconds":elapsed,"peak_gpu_memory_gb":peak_gb,
        "artifact_path":str(final),"prompt_profile":prompt_profile,"target_format":target_format,
        "effective_target_format":"original_notebook_strict" if args.task=="relation" and prompt_profile in {"text_only_strict","entity_aware_strict"} else target_format,
        "lora_r":lora_r,"lora_alpha":lora_alpha,"lora_dropout":lora_dropout,
        "focal_alpha":args.focal_alpha if strategy.focal else None,"focal_gamma":args.focal_gamma if strategy.focal else None,
        "warmup_ratio":args.warmup_ratio,"save_strategy":args.save_strategy,"save_steps":args.save_steps,
        "eval_strategy":args.eval_strategy,"eval_steps":args.eval_steps,"lr_scheduler_type":args.lr_scheduler_type,
        "max_grad_norm":args.max_grad_norm,"save_total_limit":args.save_total_limit,
        "eval_losses":loss_cb.eval_losses,"eval_history":loss_cb.history,
        "best_step":trainer.state.best_global_step,"best_eval_loss":trainer.state.best_metric,
        "best_checkpoint":trainer.state.best_model_checkpoint,
        "best_epoch":next((x["epoch"] for x in loss_cb.history if x["step"]==trainer.state.best_global_step),None),
        "best_eval_token_acc":next((x["eval_token_acc"] for x in loss_cb.history if x["step"]==trainer.state.best_global_step),None),
        **params,"trainer_metrics":train_result.metrics,
    }
    (out/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    print(f"\nTraining completed. Saved to: {final}")
    print(json.dumps({k:v for k,v in report.items() if k!="trainer_metrics"},ensure_ascii=False,indent=2))
    return report
