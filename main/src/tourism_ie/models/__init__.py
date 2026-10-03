# File: src/tourism_ie/models/__init__.py
# Module responsibility: model registry subpackage initialization file. The specific model list is in registry.py.
# Main data flow: serves as the tourism_ie.models namespace entry point.
# Reading suggestion: start with the public functions/classes in this file, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments, without modifying existing expressions, control flow, parameter defaults, or function call relationships.

from .registry import ModelSpec, MODEL_REGISTRY, resolve_model, list_models, core_model_keys, extended_model_keys
__all__ = ["ModelSpec", "MODEL_REGISTRY", "resolve_model", "list_models", "core_model_keys", "extended_model_keys"]
