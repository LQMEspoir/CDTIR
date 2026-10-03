# File: src/tourism_ie/strategies/registry.py
# Module responsibility: training strategy registry. Uses StrategySpec to describe key switches for strategies such as full/LoRA/QLoRA/focal/router.
# Main data flow: strategy key -> StrategySpec, for unified interpretation by the CLI and training modules.
# Reading suggestion: start with the public functions/classes in this file, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments, without modifying existing expressions, control flow, parameter defaults, or function call relationships.

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class StrategySpec:
    """[Class description] StrategySpec
    - Responsibility: immutable/structured spec for a training strategy, describing whether to use PEFT, 4-bit, balanced sampling, Focal Loss, or a special Router route.
    - Inheritance: does not explicitly inherit a business base class.
    - Usage: instantiated and then used by the training/inference main flow via its public methods; internal state is determined by __init__ and the saved config together.
    """
    key: str
    trainable: bool
    peft: bool
    quantized: bool = False
    focal: bool = False
    balanced: bool = False
    router: bool = False
    description: str = ""

STRATEGIES = {
    "zero_shot": StrategySpec("zero_shot", False, False, description="No fine-tuning; evaluate the base instruction model."),
    "full": StrategySpec("full", True, False, description="Full-parameter supervised fine-tuning."),
    "lora": StrategySpec("lora", True, True, description="LoRA parameter-efficient fine-tuning."),
    "qlora": StrategySpec("qlora", True, True, quantized=True, description="4-bit QLoRA fine-tuning (bitsandbytes + CUDA recommended)."),
    "balanced_lora": StrategySpec("balanced_lora", True, True, balanced=True, description="LoRA with deterministic rare-class oversampling."),
    "focal_lora": StrategySpec("focal_lora", True, True, focal=True, description="LoRA with original-compatible token-level Focal Loss alpha*(1-pt)^gamma*CE."),
    "router_multilora": StrategySpec("router_multilora", True, True, router=True,
        description="Original-compatible neural hidden-state Router + one LoRA per label + weighted-logit mixture + joint BCE router loss."),
    "grouped_router_multilora": StrategySpec("grouped_router_multilora", True, True, router=True,
        description="Previous stable grouped adapters + TF-IDF logistic router retained for backward compatibility."),
    "relation_legacy_fixed": StrategySpec("relation_legacy_fixed", True, True,
        description="Legacy relation.py reproduction: SPO offsets + Qwen2-1.5B + q/v LoRA, with causal-label alignment fixed."),
}


def get_strategy(key: str) -> StrategySpec:
    """[Function description] get_strategy
    - Responsibility: looks up a training strategy definition by string key, and raises a clear exception for unknown strategies.
    - Main parameters: key: str.
    - Returns: return type annotated as `StrategySpec`; see the implementation below for field meanings.
    - Note: includes explicit parameter/data validation; invalid input raises an exception.
    """
    try:
        return STRATEGIES[key]
    except KeyError as e:
        raise ValueError(f"Unknown strategy {key!r}. Choices: {', '.join(STRATEGIES)}") from e
