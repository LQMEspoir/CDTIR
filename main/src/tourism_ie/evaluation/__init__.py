# File: src/tourism_ie/evaluation/__init__.py
# Module responsibility: unified evaluation subpackage initialization file, used to organize evaluation components such as labels, metrics, and io.
# Main data flow: serves as the evaluation namespace entry point and does not directly execute metric computation.
# Reading suggestion: start with the public functions/classes in this file, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments, without modifying existing expressions, control flow, parameter defaults, or function call relationships.

from .metrics import evaluate_records_detailed
from .io import save_metrics
__all__ = ["evaluate_records_detailed", "save_metrics"]
