# File: scripts/benchmark_quick.py
# Module responsibility: quick benchmark launcher: a short CLI entry that calls the main benchmark flow.
# Main data flow: script entry -> delegates to the run.py benchmark subcommand.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

"""Convenience wrapper: show the 8-model LoRA benchmark matrix without loading weights."""
import subprocess, sys
raise SystemExit(subprocess.call([sys.executable,"run.py","benchmark","--task","both","--models","all","--strategies","lora","--dry-run"]))
