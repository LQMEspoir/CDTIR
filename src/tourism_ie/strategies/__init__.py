# File: src/tourism_ie/strategies/__init__.py
# Module responsibility: training strategy subpackage initialization file. Strategy definitions and resolution are in registry.py.
# Main data flow: serves as the tourism_ie.strategies namespace entry point.
# Reading suggestion: start with the public functions/classes in this file, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments, without modifying existing expressions, control flow, parameter defaults, or function call relationships.

from .registry import STRATEGIES, StrategySpec, get_strategy
__all__ = ["STRATEGIES", "StrategySpec", "get_strategy"]
