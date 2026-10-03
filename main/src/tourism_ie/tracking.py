# File: src/tourism_ie/tracking.py
# Module responsibility: experiment tracking and visualization helpers: Trainer callbacks, SwanLab init/finish, and prediction-sample logging.
# Main data flow: Trainer lifecycle / experiment config -> record eval loss, training logs, prediction samples -> SwanLab or local output.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

from __future__ import annotations
from pathlib import Path
import csv
import json
from transformers import TrainerCallback


class EvalLossCurveCallback(TrainerCallback):
    """- Responsibility: Transformers Trainer callback that records eval loss locally and outputs curve data at training end.
    - Inherits: TrainerCallback.
    - Usage: instantiated then invoked by the training/inference main flow via public methods; internal state is determined by __init__ and saved config.
    """

    def __init__(self):
        """- Responsibility: initializes object state, saving config and dependencies needed by later training/routing/loss/callback stages.
        - Parameters: no explicit business params (possibly self/cls).
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        self.eval_losses: list[float] = []
        self.history: list[dict] = []

    def on_init_end(self, args, state, control, **kwargs):
        """-- HuggingFace Trainer callback hooks --
        - Responsibility: prepares the local curve output location or callback state when Trainer init ends.
        - Parameters: args,state,control,**kwargs.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step <= 0:
            return
        path=Path(args.output_dir)/"eval_history.jsonl"
        if path.exists():
            loaded=[]
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip(): loaded.append(json.loads(line))
            by_step={int(x["step"]):x for x in loaded}
            self.history=[by_step[k] for k in sorted(by_step)]
            self.eval_losses=[float(x["eval_loss"]) for x in self.history]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """- Responsibility: extracts eval_loss and global_step after each Trainer.evaluate and appends them to the eval-loss curve data.
        - Parameters: args,state,control,metrics=None,**kwargs.
        - Returns: no explicit return; produces effects via object state, files, logs, or external training.
        - Notes: mainly in-memory; no side effects beyond the called object itself.
        """
        if metrics and "eval_loss" in metrics:
            row={"epoch":float(state.epoch or 0.0),"step":int(state.global_step),
                 "eval_loss":float(metrics["eval_loss"]),
                 "eval_token_acc":float(metrics.get("eval_token_acc",0.0))}
            self.history=[x for x in self.history if int(x["step"])!=int(row["step"])]
            self.history.append(row); self.history.sort(key=lambda x:int(x["step"]))
            self.eval_losses=[float(x["eval_loss"]) for x in self.history]
            out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
            with (out/"eval_history.jsonl").open("w",encoding="utf-8",newline="\n") as f:
                for item in self.history: f.write(json.dumps(item,ensure_ascii=False)+"\n")
            with (out/"eval_history.csv").open("w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=["epoch","step","eval_loss","eval_token_acc"])
                w.writeheader(); w.writerows(self.history)

    def on_train_end(self, args, state, control, **kwargs):
        """- Responsibility: writes/plots the accumulated eval-loss curve at training end, keeping a reproducible local trace.
        - Parameters: args,state,control,**kwargs.
        - Returns: explicitly returns None, for flow control or side effects only.
        - Notes: may read/write disk files/model weights; ensure the output directory is writable;emits logs or warnings.
        """
        if not self.eval_losses:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig = plt.figure(figsize=(8, 5))
        plt.plot([x["epoch"] for x in self.history], self.eval_losses, marker="o", linewidth=1.5)
        plt.title("Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Eval Loss")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "eval_loss_curve.png", dpi=180)
        plt.close(fig)
        print(f"Validation loss curve saved to: {out / 'eval_loss_curve.png'}")


def build_swanlab_callbacks(args, config: dict):
    """- Responsibility: builds the SwanLab/local callback list from CLI switches, degrading gracefully when dependencies are missing.
    - Parameters: args,config: dict.
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: includes explicit parameter/data validation; raises on invalid input.
    """
    if not getattr(args, "swanlab", False):
        return []
    try:
        try:
            from swanlab.integration.transformers import SwanLabCallback
        except ImportError:
            from swanlab.integration.huggingface import SwanLabCallback
    except ImportError as e:
        raise SystemExit("SwanLab tracking requested. Install optional dependency: pip install swanlab") from e
    project=getattr(args,"swanlab_project",None) or "Tourism-IE"
    experiment=getattr(args,"swanlab_experiment",None) or f"{config.get('model_key','model')}-{config.get('task','task')}-{config.get('strategy','train')}"
    cb=SwanLabCallback(
        project=project,
        experiment_name=experiment,
        description="Tourism IE training experiment",
        config=config,
    )
    return [cb]


def finish_swanlab(enabled: bool):
    """- Responsibility: safely closes the SwanLab run at experiment end, flushing uncommitted logs.
    - Parameters: enabled: bool.
    - Returns: explicitly returns None, for flow control or side effects only.
    - Notes: mainly in-memory; no side effects beyond the called object itself.
    """
    if not enabled:
        return
    try:
        import swanlab
        swanlab.finish()
    except Exception:
        pass


def log_prediction_samples(args, records: list[dict]):
    """- Responsibility: logs a few prediction samples with gold to the tracking platform for qualitative error inspection.
    - Parameters: args,records: list[dict].
    - Returns: explicitly returns None, for flow control or side effects only.
    - Notes: includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    if not getattr(args,"swanlab",False):
        return
    try:
        import swanlab
    except ImportError as e:
        raise SystemExit("SwanLab tracking requested. Install optional dependency: pip install swanlab") from e
    project=getattr(args,"swanlab_project",None) or "Tourism-IE-Prediction"
    experiment=getattr(args,"swanlab_experiment",None) or f"{getattr(args,'task','task')}-prediction"
    try:
        swanlab.init(project=project, experiment_name=experiment)
        samples=[]
        for r in records[: min(50,len(records))]:
            text=f"Input: {r.get('input','')}\nGold: {r.get('gold','')}\nPrediction: {r.get('prediction','')}\nRaw: {r.get('raw_prediction','')}"
            try: samples.append(swanlab.Text(text, caption=str(r.get('raw_prediction',''))[:200]))
            except Exception: samples.append(text)
        swanlab.log({"Prediction":samples})
    finally:
        finish_swanlab(True)
