# File: src/tourism_ie/router_multilora.py
# Module responsibility: Core implementation of the neural Router-MultiLoRA. The final-layer hidden state passes through the Router to produce category weights, then dynamically weights the token logits of multiple named LoRA adapters, jointly optimizing the language model loss and the Router BCE loss.
# Main data flow: token ids -> base hidden mean pooling -> Router logits/weights -> select category adapter -> mixture of branch logits -> CE + λ*BCE; after training the router and adapters are saved separately.
# Reading suggestions: Start with the public functions/classes in this file, then follow the imports to shared modules such as data_utils, modeling, and evaluation.
# Maintenance notes: This version only adds explanatory comments; it does not modify any existing expressions, control flow, parameter defaults, or function call relationships.

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_utils import read_jsonl, task_paths, row_labels, messages_for, render_prompt, is_entity_aware_profile, require_entity_aware_rows
from .modeling import load_tokenizer, load_base_model, find_lora_targets, count_parameters
from .models import resolve_model
from .prompts import NER_LABELS, RELATION_LABELS

ROUTER_GROUPS = {
    "ner": {
        "location": {"目的地", "出发地", "返程地"},
        # "location": {"Destination", "Origin", "Return"}
        "time": {"时长", "时间"},
        # "time": {"Duration", "Time"}
        "service": {"餐饮", "住宿", "产品", "交通"},
        # "service": {"Dining", "Lodging", "Product", "Transport"}
        "attribute": {"预算", "人数", "人群", "强度", "人气", "天气"},
        # "attribute": {"Budget", "People", "Crowd", "Intensity", "Popularity", "Weather"}
    },
    "relation": {x: {x} for x in RELATION_LABELS},
}


def _categories(task: str) -> list[str]:
    """- Responsibility: Return the fine-grained category list used by the neural Router for the given task, ensuring a fixed order between training and inference.
    - Main parameters: task: str.
    - Returns: The return type annotation is `list[str]`; see the implementation below for specific field meanings.
    - Note: This function mainly performs in-memory computation; apart from the behavior of the called objects themselves, it has no additional persistence side effects.
    """
    return list(ROUTER_GROUPS[task])


class SimpleCategoryRouter(nn.Module):
    """- Responsibility: A lightweight neural router that maps the pooled hidden state to an independent Router logit for each entity/relation category.
    - Inherits from: nn.Module.
    - Usage: After instantiation, the training/inference main pipeline calls its public methods; internal state is determined by __init__ and the saved configuration.
    """
    """Two-layer MLP router matching entity_router_rola.py.

    Original algorithm:
      hidden.mean(dim=1) -> Linear(H,H/2) -> ReLU -> Linear(H,C)
      sigmoid(logits) gives one weight per LoRA category.
    """
    def __init__(self, hidden_size: int, num_categories: int):
        """- Responsibility: initializes object state, saving config and dependencies needed by later training/routing/loss/callback stages.
        - Parameters: hidden_size: int,num_categories: int.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_size // 2), num_categories),
        )

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """- Responsibility: defines the PyTorch module forward path; input tensors flow through the module and return logits, loss, hidden state or a wrapped output.
        - Parameters: hidden: torch.Tensor.
        - Returns: return type is `torch.Tensor`; see implementation below for field details.
        - Key process: keeps standard PyTorch/Transformers calling conventions; each branch computes routing, adapter logits, task loss or test output per the class role.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.linear(pooled)


class NeuralRouterMultiLoRAModel(nn.Module):
    """- Responsibility: The current neural Router-MultiLoRA training wrapper. Shares a single PEFT base, computes multiple branches through named adapters, and mixes them by Router weights.
    - Inherits from: nn.Module.
    - Usage: After instantiation, the training/inference main pipeline calls its public methods; internal state is determined by __init__ and the saved configuration.
    """
    """Engineering-safe implementation of the original hidden-state Router-MultiLoRA.

    Differences from the broken legacy script are structural, not conceptual:
    - one PEFT model owns named adapters instead of repeatedly wrapping the same base;
    - all adapters are registered before Trainer creates its optimizer;
    - optional checkpointing recomputes the expensive 15-adapter mixture in backward;
    - base weights remain frozen as intended by LoRA.

    The training objective remains:
      final_logits = sum_i sigmoid(router_i) * logits_i
      loss = CE(final_logits, labels) + router_loss_weight * BCE(router_logits, targets)
    """
    main_input_name = "input_ids"

    def __init__(
        self,
        peft_model,
        router: SimpleCategoryRouter,
        categories: list[str],
        router_loss_weight: float = 0.5,
        forward_top_k: int = 0,
        checkpoint_mixture: bool = True,
        hidden_mode: str = "frozen_base",
    ):
        """- Responsibility: initializes object state, saving config and dependencies needed by later training/routing/loss/callback stages.
        - Parameters: peft_model,router: SimpleCategoryRouter,categories: list[str],router_loss_weight: float=0.5,forward_top_k: int=0,checkpoint_mixture: bool=True,hidden_mode: str='strict_original'.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        super().__init__()
        self.peft_model = peft_model
        self.router = router
        self.categories = list(categories)
        self.router_loss_weight = float(router_loss_weight)
        self.forward_top_k = int(forward_top_k)
        self.checkpoint_mixture = bool(checkpoint_mixture)
        if hidden_mode not in {"frozen_base", "strict_original"}:
            raise ValueError(f"Unknown router hidden mode: {hidden_mode}")
        self.hidden_mode = hidden_mode
        self._enable_all_adapter_params()

    @property
    def config(self):
        """- Responsibility: exposes the underlying model config to Hugging Face Trainer, keeping the wrapper compatible with the PreTrainedModel interface.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        return self.peft_model.config

    def _enable_all_adapter_params(self):
        """- Responsibility: re-marks all named LoRA adapter parameters trainable so a single branch isn't the only one optimized after set_adapter.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        # PEFT set_adapter toggles trainability. Trainer must see every LoRA parameter
        # as trainable when the optimizer is created, and all must be re-enabled before
        # backward so gradients from every branch can accumulate.
        for name, p in self.peft_model.named_parameters():
            if "lora_" in name or ".modules_to_save." in name:
                p.requires_grad_(True)

    def gradient_checkpointing_enable(self, **kwargs):
        """- Responsibility: forwards gradient-checkpoint enabling to the underlying model, trading compute for memory to lower peak training memory.
        - Parameters: **kwargs.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        if hasattr(self.peft_model, "gradient_checkpointing_enable"):
            self.peft_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """- Responsibility: disables the underlying model's gradient checkpointing, restoring the normal forward/backward path.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        if hasattr(self.peft_model, "gradient_checkpointing_disable"):
            self.peft_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        """- Responsibility: ensures the model input embedding path keeps gradients, for PEFT + gradient-checkpointing joint training.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        if hasattr(self.peft_model, "enable_input_require_grads"):
            self.peft_model.enable_input_require_grads()

    def _base_hidden(self, input_ids, attention_mask):
        """- Responsibility: extracts the last-layer hidden state on the adapter-disabled base path and runs mask-aware mean pooling for the router.
        - Parameters: input_ids,attention_mask.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        """Hidden states for the router.

        frozen_base: engineering default. Disable LoRA and stop gradients to save memory.
        strict_original: match entity_router_rola.py semantics by taking hidden states
        through adapter 0 with autograd enabled, so router BCE gradients can flow into
        the first LoRA adapter through the hidden-state path.
        """
        if self.hidden_mode == "strict_original":
            first = self.categories[0]
            self.peft_model.set_adapter(first)
            # Calling base_model mirrors lora_modules[first].base_model(...) from the
            # legacy script while retaining the safe named-adapter ownership model.
            outputs = self.peft_model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            self._enable_all_adapter_params()
            return outputs.hidden_states[-1]

        with torch.no_grad():
            try:
                ctx = self.peft_model.disable_adapter()
            except Exception:
                ctx = None
            if ctx is None:
                outputs = self.peft_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True, use_cache=False,
                )
            else:
                with ctx:
                    outputs = self.peft_model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, return_dict=True, use_cache=False,
                    )
        return outputs.hidden_states[-1].detach()

    def _selected_categories(self, router_weights: torch.Tensor):
        """- Responsibility: decides which adapter branches a batch actually runs, from router weights, supervised targets and forward_top_k.
        - Parameters: router_weights: torch.Tensor,router_targets: torch.Tensor | None.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: inference/tensor paths must keep device and dtype consistent.
        """
        """Select compute branches from predicted router weights only.

        ``router_targets`` are supervision for BCE and must never influence the
        forward path; using them here would make training choose branches with
        information that is unavailable at inference time.
        """
        if self.forward_top_k <= 0 or self.forward_top_k >= len(self.categories):
            return list(range(len(self.categories)))
        k=max(1,min(self.forward_top_k,len(self.categories)))
        selected=set(torch.topk(router_weights.detach(),k=k,dim=-1).indices.flatten().tolist())
        return sorted(int(x) for x in selected)

    def _mixture_logits(self, input_ids, attention_mask, router_weights, selected_indices: Iterable[int]):
        """- Responsibility: activates each selected named adapter to get token logits, then differentiably fuses them by per-sample router weights.
        - Parameters: input_ids,attention_mask,router_weights,selected_indices: Iterable[int].
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Key process: only run selected_indices branches to avoid wasted compute; each branch is activated via set_adapter; router weights are broadcast over batch samples to token/vocab dims then summed.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        final_logits=None
        for i in selected_indices:
            name=self.categories[i]

            def branch(ids, mask, weight, adapter_name=name):
                self.peft_model.set_adapter(adapter_name)
                out=self.peft_model(
                    input_ids=ids,
                    attention_mask=mask,
                    return_dict=True,
                    use_cache=False,
                )
                return out.logits * weight.unsqueeze(-1).unsqueeze(-1)

            if self.checkpoint_mixture and self.training and torch.is_grad_enabled():
                # Checkpoint each adapter branch separately. Checkpointing the whole
                # mutable-adapter loop causes only the final adapter to receive grads.
                from torch.utils.checkpoint import checkpoint
                weighted=checkpoint(branch,input_ids,attention_mask,router_weights[:,i],use_reentrant=False)
            else:
                weighted=branch(input_ids,attention_mask,router_weights[:,i])
            final_logits=weighted if final_logits is None else final_logits + weighted
        self._enable_all_adapter_params()
        if final_logits is None:
            raise RuntimeError("Router-MultiLoRA selected no adapter branches")
        return final_logits

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        router_targets=None,
        router_input_ids=None,
        router_attention_mask=None,
        **kwargs,
    ):
        """- Responsibility: defines the PyTorch module forward path; input tensors flow through the module and return logits, loss, hidden state or a wrapped output.
        - Parameters: input_ids,attention_mask=None,labels=None,router_targets=None,**kwargs.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Key process: keeps standard PyTorch/Transformers calling conventions; each branch computes routing, adapter logits, task loss or test output per the class role.
        - Notes: inference/tensor paths must keep device and dtype consistent;emits logs or warnings.
        """
        # Router features are computed from a separately encoded prompt-only
        # sequence.  The causal-LM input still contains Prompt+Gold Target for
        # teacher forcing, but those target tokens are never passed to Router.
        if router_input_ids is None:
            router_input_ids=input_ids
            router_attention_mask=attention_mask
            if labels is not None:
                prompt_mask=labels.eq(-100)
                if attention_mask is not None:
                    prompt_mask=prompt_mask & attention_mask.bool()
                pad_id=getattr(self.config,"pad_token_id",None)
                if pad_id is None: pad_id=0
                router_input_ids=input_ids.masked_fill(~prompt_mask,int(pad_id))
                router_attention_mask=prompt_mask.to(dtype=attention_mask.dtype if attention_mask is not None else torch.long)
        elif router_attention_mask is None:
            router_attention_mask=torch.ones_like(router_input_ids)

        hidden=self._base_hidden(router_input_ids,router_attention_mask)
        # Match original dtype/device placement of the router to base hidden.
        if next(self.router.parameters()).device != hidden.device:
            self.router.to(hidden.device)
        if next(self.router.parameters()).dtype != hidden.dtype:
            self.router.to(dtype=hidden.dtype)
        router_logits=self.router(hidden, router_attention_mask)
        router_weights=torch.sigmoid(router_logits)
        if router_targets is not None and not torch.is_tensor(router_targets):
            router_targets=torch.tensor(router_targets,device=router_logits.device)
        if router_targets is not None:
            router_targets=router_targets.to(device=router_logits.device,dtype=router_logits.dtype)
        selected=self._selected_categories(router_weights)

        final_logits=self._mixture_logits(input_ids,attention_mask,router_weights,selected)

        loss=None
        if labels is not None:
            shift_logits=final_logits[..., :-1, :].contiguous()
            shift_labels=labels[..., 1:].contiguous()
            ce=F.cross_entropy(
                shift_logits.view(-1,shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss=ce
            if router_targets is not None:
                router_loss=F.binary_cross_entropy_with_logits(router_logits,router_targets)
                loss=loss + self.router_loss_weight*router_loss
        return {"loss":loss,"logits":final_logits,"router_logits":router_logits} if loss is not None else {"logits":final_logits,"router_logits":router_logits}


class RouterDataCollator:
    """- Responsibility: A data collator dedicated to Router training; in addition to standard causal-LM padding, it batches the multi-label router_targets together.
    - Inherits from: no explicit business base class.
    - Usage: After instantiation, the training/inference main pipeline calls its public methods; internal state is determined by __init__ and the saved configuration.
    """
    def __init__(self, tokenizer):
        """- Responsibility: initializes object state, saving config and dependencies needed by later training/routing/loss/callback stages.
        - Parameters: tokenizer.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: inference/tensor paths must keep device and dtype consistent.
        """
        from transformers import DataCollatorForSeq2Seq
        self.tokenizer=tokenizer
        self.base=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100)

    def __call__(self, features):
        """- Responsibility: pads variable-length training samples into batch tensors and stacks the router multi-label supervision vector.
        - Parameters: features.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        targets=[f.pop("router_targets") for f in features]
        router_ids=[f.pop("router_input_ids") for f in features]
        router_masks=[f.pop("router_attention_mask") for f in features]
        batch=self.base(features)
        max_len=max(len(x) for x in router_ids)
        pad_id=self.tokenizer.pad_token_id
        if pad_id is None: pad_id=self.tokenizer.eos_token_id
        if pad_id is None: pad_id=0
        padded_ids=[]; padded_masks=[]
        for ids,mask in zip(router_ids,router_masks):
            pad=max_len-len(ids)
            if self.tokenizer.padding_side=="left":
                padded_ids.append([pad_id]*pad+ids)
                padded_masks.append([0]*pad+mask)
            else:
                padded_ids.append(ids+[pad_id]*pad)
                padded_masks.append(mask+[0]*pad)
        batch["router_input_ids"]=torch.tensor(padded_ids,dtype=torch.long)
        batch["router_attention_mask"]=torch.tensor(padded_masks,dtype=torch.long)
        batch["router_targets"]=torch.tensor(targets,dtype=torch.float32)
        return batch


def _encode_router_dataset(tokenizer, task, rows, max_length, prompt_profile="canonical", target_format="canonical"):
    """- Responsibility: Encode raw samples into causal-LM tokens/labels while generating each sample's multi-hot router_targets.
    - Main parameters: tokenizer, task, rows, max_length, prompt_profile='canonical', target_format='canonical'.
    - Returns: Returns the processed object/metric/tensor/path, etc.; the actual structure is determined by each return branch.
    - Note: The model inference/tensor path must keep device and dtype consistent.
    """
    from datasets import Dataset
    from .data_utils import target_text
    cats=_categories(task)
    index={x:i for i,x in enumerate(cats)}
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
            a=a+[tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id]
        elif tokenizer.eos_token_id is not None:
            a=a+[tokenizer.eos_token_id]
        prompt_ids=p[:max_length]
        ids=(p+a)[:max_length]
        labels=([-100]*len(p)+a)[:max_length]
        target=[0.0]*len(cats)
        item_labels=row_labels(task,row)
        for group, members in ROUTER_GROUPS[task].items():
            if item_labels & members:
                target[index[group]]=1.0
        return {
            "input_ids":ids,
            "attention_mask":[1]*len(ids),
            "labels":labels,
            "router_input_ids":prompt_ids,
            "router_attention_mask":[1]*len(prompt_ids),
            "router_targets":target,
        }
    return Dataset.from_list(rows).map(encode,remove_columns=list(rows[0].keys()))


def _locate_saved_adapter(root: Path, adapter_name: str) -> Path:
    """- Responsibility: Locate the actually saved PEFT adapter subdirectory within the training output directory, compatible with different Trainer save layouts.
    - Main parameters: root: Path, adapter_name: str.
    - Returns: The return type annotation is `Path`; see the implementation below for specific field meanings.
    - Note: Contains explicit parameter/data validation; invalid input raises an exception.
    """
    if (root/"adapter_config.json").exists(): return root
    exact=[p.parent for p in root.rglob("adapter_config.json") if p.parent.name==adapter_name]
    if exact: return exact[0]
    found=list(root.rglob("adapter_config.json"))
    if not found: raise RuntimeError(f"No adapter_config.json saved for {adapter_name} under {root}")
    return found[0].parent


def _save_adapters(peft_model, categories: list[str], adapters_root: Path):
    """- Responsibility: Save the multiple named adapters of a PEFT model into separate directories, so they can be restored by name at inference time.
    - Main parameters: peft_model, categories: list[str], adapters_root: Path.
    - Returns: Returns the processed object/metric/tensor/path, etc.; the actual structure is determined by each return branch.
    - Note: May read/write disk files or model weights; ensure the output directory is writable. Involves training or backpropagation; GPU memory/randomness affects resource usage and reproducibility.
    """
    adapters_root.mkdir(parents=True,exist_ok=True)
    saved={}
    for name in categories:
        tmp=adapters_root/f"_save_{name}"
        peft_model.save_pretrained(tmp,selected_adapters=[name])
        actual=_locate_saved_adapter(tmp,name)
        final=adapters_root/name
        final.mkdir(parents=True,exist_ok=True)
        for p in actual.iterdir():
            if p.is_file(): shutil.copy2(p,final/p.name)
        shutil.rmtree(tmp,ignore_errors=True)
        saved[name]=str(Path("adapters")/name)
    return saved


def train_router_multilora(root: Path, args):
    """- Responsibility: Train the Router-MultiLoRA package: prepare category/group data, train multiple adapters and the Router, and save a unified manifest.
    - Main parameters: root: Path, args.
    - Returns: Returns the processed object/metric/tensor/path, etc.; the actual structure is determined by each return branch.
    - Key process: Establish category/group adapters; train the supervision required by the Router; save each adapter, the Router weights, and the manifest so everything can be fully restored later.
    - Note: May read/write disk files or model weights; ensure the output directory is writable. Involves training or backpropagation; GPU memory/randomness affects resource usage and reproducibility. The model inference/tensor path must keep device and dtype consistent. Contains explicit parameter/data validation; invalid input raises an exception. May emit log or warning messages.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import TrainingArguments, Trainer, set_seed
    except ImportError as e:
        raise SystemExit("Missing training dependencies. Run: pip install -r requirements.txt") from e
    from .training import _effective_lora_hparams, _compute_token_metrics, _argmax_logits, _resume_value
    from .tracking import build_swanlab_callbacks, finish_swanlab, EvalLossCurveCallback

    set_seed(args.seed)
    prompt_profile=getattr(args,"prompt_profile","canonical")
    train_path,eval_path=task_paths(root,args.task,args.split,prompt_profile,getattr(args,"data_dir",None))
    train_rows=read_jsonl(train_path); eval_rows=read_jsonl(eval_path)
    require_entity_aware_rows(train_rows,prompt_profile,"router training data")
    require_entity_aware_rows(eval_rows,prompt_profile,"router evaluation data")
    if args.limit:
        train_rows=train_rows[:args.limit]
        eval_rows=eval_rows[:max(1,min(args.limit//5 or 1,len(eval_rows)))]

    spec=resolve_model(args.model)
    tok=load_tokenizer(args.model)
    base=load_base_model(args.model,training=True,quantized=False)
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        if hasattr(base,"enable_input_require_grads"): base.enable_input_require_grads()
    targets=find_lora_targets(base,spec.target_modules)
    r,alpha,dropout=_effective_lora_hparams(args)
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,inference_mode=False,r=r,lora_alpha=alpha,lora_dropout=dropout,target_modules=targets)
    cats=_categories(args.task)
    peft_model=get_peft_model(base,lora,adapter_name=cats[0])
    for name in cats[1:]: peft_model.add_adapter(name,lora)
    for _,p in peft_model.named_parameters():
        if "lora_" in _: p.requires_grad_(True)

    hidden_size=int(getattr(base.config,"hidden_size",getattr(base.config,"n_embd",0)))
    if hidden_size<=0: raise RuntimeError("Could not determine hidden_size for neural router")
    router=SimpleCategoryRouter(hidden_size,len(cats))
    try:
        pdtype=next(peft_model.parameters()).dtype
        router.to(dtype=pdtype)
    except StopIteration: pass
    model=NeuralRouterMultiLoRAModel(
        peft_model,router,cats,
        router_loss_weight=getattr(args,"router_loss_weight",0.5),
        forward_top_k=getattr(args,"router_forward_top_k",0),
        checkpoint_mixture=getattr(args,"router_checkpoint_mixture",True),
        hidden_mode=getattr(args,"router_hidden_mode","frozen_base"),
    )

    target_format=getattr(args,"target_format","canonical")
    train_ds=_encode_router_dataset(tok,args.task,train_rows,args.max_length,prompt_profile,target_format)
    eval_ds=_encode_router_dataset(tok,args.task,eval_rows,args.max_length,prompt_profile,target_format)
    out=Path(args.output_dir) if args.output_dir else root/"outputs"/f"{args.task}_{spec.key}_router_multilora"
    out.mkdir(parents=True,exist_ok=True)

    use_cuda=torch.cuda.is_available(); bf16=use_cuda and torch.cuda.is_bf16_supported(); fp16=use_cuda and not bf16
    if use_cuda: torch.cuda.reset_peak_memory_stats()
    targs=TrainingArguments(
        output_dir=str(out/"trainer"),num_train_epochs=args.epochs,
        max_steps=getattr(args,"max_steps",-1),
        per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,warmup_ratio=args.warmup_ratio,logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,eval_steps=args.eval_steps,
        lr_scheduler_type=args.lr_scheduler_type,max_grad_norm=args.max_grad_norm,
        save_total_limit=args.save_total_limit,report_to="none",remove_unused_columns=False,
        load_best_model_at_end=(args.save_strategy==args.eval_strategy and args.save_strategy!="no"),
        metric_for_best_model="eval_loss",greater_is_better=False,
        gradient_checkpointing=args.gradient_checkpointing,bf16=bf16,fp16=fp16,
        dataloader_pin_memory=use_cuda,seed=args.seed,data_seed=args.seed,
    )
    loss_cb=EvalLossCurveCallback()
    callbacks=[loss_cb]+build_swanlab_callbacks(args,{"task":args.task,"model_key":spec.key,"model_name":spec.model_name,
        "strategy":"router_multilora","router_categories":cats,"router_loss_weight":model.router_loss_weight,
        "router_forward_top_k":model.forward_top_k,"router_hidden_mode":model.hidden_mode,"lora_r":r,"lora_alpha":alpha,"lora_dropout":dropout,
        "prompt_profile":prompt_profile,"target_format":target_format})
    class LightweightRouterTrainer(Trainer):
        """Checkpoint trainable Router/LoRA state only; base Qwen remains external."""
        def save_model(self, output_dir=None, _internal_call=False):
            target=Path(output_dir or self.args.output_dir); target.mkdir(parents=True,exist_ok=True)
            state={k:v.detach().cpu() for k,v in self.model.state_dict().items()
                   if k.startswith("router.") or "lora_" in k or ".modules_to_save." in k}
            torch.save(state,target/"router_lora_state.pt")
            tok.save_pretrained(target/"tokenizer")
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            state_path=Path(resume_from_checkpoint)/"router_lora_state.pt"
            if not state_path.exists():
                raise ValueError(f"Missing lightweight Router checkpoint: {state_path}")
            try: state=torch.load(state_path,map_location="cpu",weights_only=True)
            except TypeError: state=torch.load(state_path,map_location="cpu")
            (model or self.model).load_state_dict(state,strict=False)
        def _load_best_model(self):
            if self.state.best_model_checkpoint:
                self._load_from_checkpoint(self.state.best_model_checkpoint,self.model)

    trainer=LightweightRouterTrainer(model=model,args=targs,train_dataset=train_ds,eval_dataset=eval_ds,
        data_collator=RouterDataCollator(tok),compute_metrics=_compute_token_metrics,
        preprocess_logits_for_metrics=_argmax_logits,callbacks=callbacks)
    start=time.perf_counter()
    try:
        result=trainer.train(resume_from_checkpoint=_resume_value(
            getattr(args,"resume_from_checkpoint",None),targs.output_dir,getattr(args,"max_steps",-1)))
    except KeyboardInterrupt:
        interrupt_dir=Path(targs.output_dir)/f"checkpoint-{trainer.state.global_step}-interrupt"
        trainer.save_model(interrupt_dir)
        trainer.state.save_to_json(str(interrupt_dir/"trainer_state.json"))
        if trainer.optimizer is not None: torch.save(trainer.optimizer.state_dict(),interrupt_dir/"optimizer.pt")
        if trainer.lr_scheduler is not None: torch.save(trainer.lr_scheduler.state_dict(),interrupt_dir/"scheduler.pt")
        if hasattr(trainer,"_save_rng_state"): trainer._save_rng_state(str(interrupt_dir))
        raise
    finally:
        finish_swanlab(getattr(args,"swanlab",False))
    elapsed=time.perf_counter()-start

    model._enable_all_adapter_params()
    adapters=_save_adapters(model.peft_model,cats,out/"adapters")
    torch.save({k:v.detach().cpu() for k,v in model.router.state_dict().items()},out/"router.pt")
    tok.save_pretrained(out/"tokenizer")
    peak=(torch.cuda.max_memory_allocated()/1024**3) if use_cuda else 0.0
    params=count_parameters(model)
    manifest={
        "format":"tourism_ie_neural_router_multilora_v2",
        "algorithm_reference":"legacy/original_scripts/entity_router_rola.py",
        "task":args.task,"model_key":spec.key,"model_name":spec.model_name,"strategy":"router_multilora",
        "categories":cats,"adapters":adapters,"router_file":"router.pt","hidden_size":hidden_size,
        "router_architecture":"Linear(H,H/2)->ReLU->Linear(H,C); attention-mask weighted mean pooling; sigmoid weights",
        "router_feature_source":"prompt_only_base_hidden_no_gold_target",
        "mixture":"sum_i sigmoid(router_i) * adapter_i_logits",
        "router_loss":"BCEWithLogitsLoss","router_loss_weight":model.router_loss_weight,
        "router_forward_top_k":model.forward_top_k,"router_checkpoint_mixture":model.checkpoint_mixture,"router_hidden_mode":model.hidden_mode,
        "prompt_profile":prompt_profile,"target_format":target_format,
        "lora_r":r,"lora_alpha":alpha,"lora_dropout":dropout,
        "warmup_ratio":args.warmup_ratio,"trainer_checkpoint_policy":"lightweight LoRA + Router + optimizer/scheduler/trainer/random states",
        "training_seconds":elapsed,"peak_gpu_memory_gb":peak,"eval_losses":loss_cb.eval_losses,"eval_history":loss_cb.history,
        "best_step":trainer.state.best_global_step,"best_eval_loss":trainer.state.best_metric,
        "best_checkpoint":trainer.state.best_model_checkpoint,
        "best_epoch":next((x["epoch"] for x in loss_cb.history if x["step"]==trainer.state.best_global_step),None),
        "best_eval_token_acc":next((x["eval_token_acc"] for x in loss_cb.history if x["step"]==trainer.state.best_global_step),None),**params,
    }
    (out/"router_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    report={**manifest,"split":args.split,"seed":args.seed,"train_rows_original":len(train_rows),"train_rows_effective":len(train_rows),
            "eval_rows":len(eval_rows),"epochs":args.epochs,"artifact_path":str(out),"trainer_metrics":result.metrics}
    (out/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    print(f"\nNeural Router-MultiLoRA training completed. Saved to: {out}")
    print(json.dumps({k:v for k,v in report.items() if k!="trainer_metrics"},ensure_ascii=False,indent=2))
    return report


def load_router_package(router_dir: str|Path, quantized: bool=False):
    """- Responsibility: Read the Router manifest, tokenizer, base/PEFT model, and Router weights from disk, and rebuild a complete package that can be used for inference directly.
    - Main parameters: router_dir: str | Path, quantized: bool=False.
    - Returns: Returns the processed object/metric/tensor/path, etc.; the actual structure is determined by each return branch.
    - Note: Involves training or backpropagation; GPU memory/randomness affects resource usage and reproducibility. The model inference/tensor path must keep device and dtype consistent. Contains explicit parameter/data validation; invalid input raises an exception.
    """
    from peft import PeftModel
    router_dir=Path(router_dir)
    manifest=json.loads((router_dir/"router_manifest.json").read_text(encoding="utf-8"))
    if not str(manifest.get("format","")).startswith("tourism_ie_neural_router_multilora"):
        raise ValueError(f"Not a neural Router-MultiLoRA package: {manifest.get('format')}")
    tok=load_tokenizer(manifest["model_name"])
    base=load_base_model(manifest["model_name"],training=False,quantized=quantized)
    cats=list(manifest["categories"]); adapters=manifest["adapters"]
    first=cats[0]
    model=PeftModel.from_pretrained(base,router_dir/adapters[first],adapter_name=first,is_trainable=False)
    for name in cats[1:]: model.load_adapter(router_dir/adapters[name],adapter_name=name,is_trainable=False)
    if not getattr(model,"hf_device_map",None):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    else:
        try: device=next(model.parameters()).device
        except StopIteration: device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router=SimpleCategoryRouter(int(manifest["hidden_size"]),len(cats))
    try:
        state=torch.load(router_dir/manifest["router_file"],map_location="cpu",weights_only=True)
    except TypeError:
        state=torch.load(router_dir/manifest["router_file"],map_location="cpu")
    router.load_state_dict(state)
    dtype=next(model.parameters()).dtype
    router.to(device=device,dtype=dtype); router.eval(); model.eval()
    return tok,model,router,manifest,device


def _prompt_hidden(text,task,tok,model,device,prompt_profile="canonical",hidden_mode="frozen_base",first_adapter=None,entities=None):
    """- Responsibility: Run only the base model's hidden-state path, apply mask-aware pooling to the input prompt, and obtain the Router inference features.
    - Main parameters: text, task, tok, model, device, prompt_profile='canonical', hidden_mode='strict_original', first_adapter=None, entities=None.
    - Returns: Returns the processed object/metric/tensor/path, etc.; the actual structure is determined by each return branch.
    - Note: The model inference/tensor path must keep device and dtype consistent.
    """
    prompt=render_prompt(tok,task,text,prompt_profile,add_generation_prompt=True,entities=entities)
    inputs=tok(prompt,return_tensors="pt")
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        if hidden_mode == "strict_original" and first_adapter is not None:
            model.set_adapter(first_adapter)
            out=model.base_model(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
        else:
            try: ctx=model.disable_adapter()
            except Exception: ctx=None
            if ctx is None:
                out=model(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
            else:
                with ctx: out=model(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
    return inputs,out.hidden_states[-1]


def route_text(text: str, tok, model, router, manifest: dict, device, threshold: float|None=None, entities=None):
    """- Responsibility: Compute Router probabilities from the input text and select the target category/semantic group, while applying a top-k or low-confidence fallback strategy.
    - Main parameters: text: str, tok, model, router, manifest: dict, device, threshold: float | None=None, entities=None.
    - Returns: Returns the processed object/metric/tensor/path, etc.; the actual structure is determined by each return branch.
    - Note: This function mainly performs in-memory computation; apart from the behavior of the called objects themselves, it has no additional persistence side effects.
    """
    # threshold is accepted for CLI compatibility. Neural multi-label routing does not
    # need a general fallback; top-1 is always one of the original category adapters.
    inputs,hidden=_prompt_hidden(text,manifest["task"],tok,model,device,manifest.get("prompt_profile","canonical"),manifest.get("router_hidden_mode","frozen_base"),manifest.get("categories",[None])[0],entities=entities)
    with torch.inference_mode(): probs=torch.sigmoid(router(hidden,inputs.get("attention_mask")))[0]
    i=int(torch.argmax(probs).item())
    return manifest["categories"][i],float(probs[i].item()),{manifest["categories"][j]:float(probs[j].item()) for j in range(len(probs))}


def generate_weighted(task: str,text: str,tok,model,router,manifest: dict,device,max_new_tokens: int=256,top_k: int=0,entities=None):
    """- Responsibility: Perform neural Router-MultiLoRA autoregressive generation: at each step fuse the multiple adapter logits according to the Router weights and select the next token.
    - Main parameters: task: str, text: str, tok, model, router, manifest: dict, device, max_new_tokens: int=256, top_k: int=0, entities=None.
    - Returns: Returns the processed object/metric/tensor/path, etc.; the actual structure is determined by each return branch.
    - Key process: The Router weights stay consistent with the input semantics throughout the whole generation process; each decoding step obtains the adapter logits separately, then selects the token after normalized weighting.
    - Note: The model inference/tensor path must keep device and dtype consistent.
    """
    """Faithful greedy generation using the weighted sum of all adapter logits.

    This is intentionally expensive (up to 15 adapter forwards per generated token),
    but it completes the inference path that the original legacy script omitted.
    Use top_k>0 to approximate the mixture when speed/memory is more important.
    """
    prompt=render_prompt(tok,task,text,manifest.get("prompt_profile","canonical"),add_generation_prompt=True,entities=entities)
    enc=tok(prompt,return_tensors="pt"); seq=enc["input_ids"].to(device); mask=enc.get("attention_mask",torch.ones_like(seq)).to(device)
    cats=manifest["categories"]
    start=time.perf_counter()
    with torch.inference_mode():
        # Match training exactly: compute routing once from the original prompt
        # only.  Generated answer tokens must not change the routing decision.
        if manifest.get("router_hidden_mode", "frozen_base") == "strict_original":
            model.set_adapter(cats[0])
            base_out=model.base_model(input_ids=seq,attention_mask=mask,output_hidden_states=True,return_dict=True,use_cache=False)
        else:
            try: ctx=model.disable_adapter()
            except Exception: ctx=None
            if ctx is None: base_out=model(input_ids=seq,attention_mask=mask,output_hidden_states=True,return_dict=True,use_cache=False)
            else:
                with ctx: base_out=model(input_ids=seq,attention_mask=mask,output_hidden_states=True,return_dict=True,use_cache=False)
        probs=torch.sigmoid(router(base_out.hidden_states[-1],mask))
        first_probs=probs[0].detach().cpu()
        if top_k and top_k<len(cats): idxs=torch.topk(probs[0],k=max(1,top_k)).indices.tolist()
        else: idxs=list(range(len(cats)))

        for _ in range(max_new_tokens):
            mixed=None
            for i in idxs:
                model.set_adapter(cats[i])
                logits=model(input_ids=seq,attention_mask=mask,return_dict=True,use_cache=False).logits[:,-1,:]
                weighted=logits*probs[:,i].unsqueeze(-1)
                mixed=weighted if mixed is None else mixed+weighted
            nxt=torch.argmax(mixed,dim=-1,keepdim=True)
            seq=torch.cat([seq,nxt],dim=1); mask=torch.cat([mask,torch.ones_like(nxt)],dim=1)
            if tok.eos_token_id is not None and bool((nxt==tok.eos_token_id).all()): break
    elapsed=time.perf_counter()-start
    gen=seq[0,enc["input_ids"].shape[1]:]
    raw=tok.decode(gen,skip_special_tokens=True).strip()
    probs_dict={cats[i]:float(first_probs[i].item()) for i in range(len(cats))}
    return raw,elapsed,probs_dict
