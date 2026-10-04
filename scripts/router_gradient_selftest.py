# File: scripts/router_gradient_selftest.py
# Module responsibility: gradient & mixture self-test for the neural Router-MultiLoRA: uses a lightweight FakePeft model to verify the router, adapter mixture and hidden-state path are backprop-able.
# Main data flow: build a fake PEFT model -> run mixture/hidden self-tests -> check gradients and output shapes -> return PASS/FAIL.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

"""Toy backward tests for neural Router-MultiLoRA.

Validates:
1) branch checkpointing sends gradients to every adapter branch and the neural router;
2) strict_original hidden mode restores the legacy router->hidden->adapter0 gradient path;
3) frozen_base hidden mode intentionally blocks that hidden-state gradient path.
"""
from __future__ import annotations
import contextlib,sys,os
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(os.environ.get("CDTIR_ROOT",Path.cwd())).expanduser().resolve()
SRC=Path(__file__).resolve().parents[1]/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from tourism_ie.router_multilora import NeuralRouterMultiLoRAModel,SimpleCategoryRouter

class FakePeft(torch.nn.Module):
    """[Class] FakePeft
    - Responsibility: lightweight PEFT stand-in for unit self-tests; mocks the named-adapter, base hidden-state and logits interfaces without real weights.
    - Inherits: torch.nn.Module.
    - Usage: instantiated then invoked by the training/inference main flow via public methods; internal state is determined by __init__ and saved config.
    """
    def __init__(self,categories,h=8,vocab=13):
        """[Function] FakePeft.__init__
        - Responsibility: initializes object state, saving config and dependencies needed by later training/routing/loss/callback stages.
        - Parameters: categories,h=8,vocab=13.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        super().__init__(); self.emb=torch.nn.Embedding(20,h); self.head=torch.nn.Linear(h,vocab)
        for p in list(self.emb.parameters())+list(self.head.parameters()): p.requires_grad=False
        self.lora_delta=torch.nn.ParameterDict({c:torch.nn.Parameter(torch.randn(h)*0.01) for c in categories})
        self.active=categories[0]; self.config=SimpleNamespace(hidden_size=h)
    @property
    def base_model(self):
        """[Function] FakePeft.base_model
        - Responsibility: returns the base-model interface held by the FakePeft/wrapper, to mimic the real PEFT API.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        return self
    def set_adapter(self,name):
        """[Function] FakePeft.set_adapter
        - Responsibility: switches the active named adapter so later forwards use only the specified LoRA branch.
        - Parameters: name.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        self.active=name
        for n,p in self.lora_delta.items(): p.requires_grad_(n==name)
    @contextlib.contextmanager
    def disable_adapter(self):
        """[Function] FakePeft.disable_adapter
        - Responsibility: temporarily disables the adapter to get pure base-model output or hidden state.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        old=self.active; self.active=None
        try: yield
        finally: self.active=old
    def forward(self,input_ids,attention_mask=None,output_hidden_states=False,return_dict=True,use_cache=False,**kwargs):
        """[Function] FakePeft.forward
        - Responsibility: defines the PyTorch module forward path; input tensors flow through the module and return logits, loss, hidden state or a wrapped output.
        - Parameters: input_ids,attention_mask=None,output_hidden_states=False,return_dict=True,use_cache=False,**kwargs.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Key process: keeps standard PyTorch/Transformers calling conventions; each branch computes routing, adapter logits, task loss or test output per the class role.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        h=self.emb(input_ids)
        if self.active is not None: h=h+self.lora_delta[self.active].view(1,1,-1)
        logits=self.head(h)
        return SimpleNamespace(logits=logits,hidden_states=[h] if output_hidden_states else None)


def mixture_test():
    """[Function] mixture_test
    - Responsibility: builds a minimal multi-adapter mixture scenario, verifying each LoRA branch's logits can be combined by router weights with gradients flowing back.
    - Parameters: no explicit business params (possibly self/cls).
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: involves training/backprop; memory/randomness affect resource usage and reproducibility;includes explicit parameter/data validation; raises on invalid input.
    """
    cats=["location","time","service","attribute"]
    peft=FakePeft(cats); router=SimpleCategoryRouter(8,len(cats))
    model=NeuralRouterMultiLoRAModel(peft,router,cats,checkpoint_mixture=True,hidden_mode="strict_original")
    ids=torch.tensor([[1,2,3,4]]); mask=torch.ones_like(ids); labels=torch.tensor([[-100,2,3,4]])
    targets=torch.tensor([[1.0,1.0,0.0,0.0]])
    loss=model(ids,mask,labels=labels,router_targets=targets)["loss"]; loss.backward()
    adapter_grads=[peft.lora_delta[c].grad is not None for c in cats]
    router_grads=[p.grad is not None for p in router.parameters()]
    if not (all(adapter_grads) and all(router_grads)):
        raise AssertionError((adapter_grads,router_grads))
    return loss.item(),adapter_grads,router_grads


def hidden_path_test(mode):
    """[Function] hidden_path_test
    - Responsibility: verifies the router's hidden-state acquisition path doesn't cut gradients, and checks tensor shapes/graph meet training requirements.
    - Parameters: mode.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: involves training/backprop; memory/randomness affect resource usage and reproducibility.
    """
    cats=["a","b","c"]
    peft=FakePeft(cats); router=SimpleCategoryRouter(8,len(cats))
    model=NeuralRouterMultiLoRAModel(peft,router,cats,checkpoint_mixture=False,hidden_mode=mode)
    ids=torch.tensor([[1,2,3,4]]); mask=torch.ones_like(ids)
    hidden=model._base_hidden(ids,mask)
    score=model.router(hidden).sum()
    score.backward()
    return [peft.lora_delta[c].grad is not None for c in cats]


def main():
    """[Function] main
    - Responsibility: script entry: parse args, run required validation, and invoke the corresponding business flow per command.
    - Parameters: no explicit business params (possibly self/cls).
    - Returns: no explicit return; produces effects via object state, files, logs, or external training.
    - Notes: includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    loss,adapter_grads,router_grads=mixture_test()
    strict=hidden_path_test("strict_original")
    frozen=hidden_path_test("frozen_base")
    print(f"mixture_loss={loss:.6f}")
    print("mixture_adapter_gradients=",adapter_grads)
    print("router_gradients=",router_grads)
    print("strict_hidden_adapter_gradients=",strict)
    print("frozen_hidden_adapter_gradients=",frozen)
    if strict != [True,False,False]:
        raise SystemExit(f"strict_original should route hidden-state gradients only to adapter0, got {strict}")
    if any(frozen):
        raise SystemExit(f"frozen_base hidden path should be detached, got {frozen}")
    print("RESULT: PASS")

if __name__=="__main__": main()
