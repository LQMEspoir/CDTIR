# File: src/tourism_ie/evaluation/labels.py
# Module responsibility: task label set query utilities, uniformly returning the category list for the NER or relation extraction task.
# Main data flow: task name -> corresponding label list.
# Reading suggestion: start with the public functions/classes in this file, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments, without modifying existing expressions, control flow, parameter defaults, or function call relationships.

from ..prompts import NER_LABELS, RELATION_LABELS

def labels_for_task(task: str):
    """[Function description] labels_for_task
    - Responsibility: returns the fixed label space of a task; unknown tasks raise an explicit error to avoid computing metrics on the wrong category set.
    - Main parameters: task: str.
    - Returns: returns the processed object/metrics/tensor/path, etc.; the actual structure depends on each return branch.
    - Note: this function mainly does in-memory computation; no extra persistence side effects beyond the called object's own behavior.
    """
    return list(NER_LABELS if task == "ner" else RELATION_LABELS)
