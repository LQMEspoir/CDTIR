#!/usr/bin/env bash
# run training/ablation in the background under tmux, so closing the terminal/VSCode doesn't interrupt it.
#
# Usage:
#   bash run_tmux.sh                  # default: run backbone (GPU 0)
#   bash run_tmux.sh ablation 1       # run ablation on GPU 1
#   bash run_tmux.sh resume 0         # resume on GPU 0
#
# arg 1: task (backbone|ablation|resume)
# arg 2: GPU id (0/1/..., default 0)
#
# backbone can also use MODELS to run only some models:
#   MODELS="glm4-9b" bash run_tmux.sh backbone 0
set -e

TASK="${1:-backbone}"
GPU="${2:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
case "$TASK" in
  backbone) SCRIPT="run_backbone.sh"; SESSION="backbone-gpu${GPU}" ;;
  ablation) SCRIPT="run_ablation.sh"; SESSION="ablation-gpu${GPU}" ;;
  resume) SCRIPT="resume.sh"; SESSION="resume-gpu${GPU}" ;;
  *) echo "usage: bash run_tmux.sh [backbone|ablation|resume] [gpu_id]"; exit 1 ;;
esac
LOG="/tmp/${SESSION}_tmux.log"

# 1) install tmux (if not installed)
if ! command -v tmux &>/dev/null; then
    echo "tmux not installed, installing..."
    apt-get update && apt-get install -y tmux
fi

# 2) attach directly if the session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' exists, attaching (Ctrl+b then d to detach)..."
    exec tmux attach -t "$SESSION"
fi

# 3) create a detached session and run the script in it (output also written to a log)
cd "$(dirname "$0")"
echo "starting '$SCRIPT' in tmux session '$SESSION' (GPU $GPU)..."
# tmux new-session doesn't inherit this script's env vars; pass them explicitly into the session
CMD="bash $SCRIPT 2>&1 | tee $LOG"
[ -n "$MODELS" ] && CMD="MODELS=\"$MODELS\" $CMD"
tmux new-session -d -s "$SESSION" "CUDA_VISIBLE_DEVICES=$GPU $CMD"

echo ""
echo "✅ started in background. common commands:"
echo "   tmux attach -t $SESSION         # attach to view live output"
echo "   tail -f $LOG                    # view log file only"
echo "   tmux ls                         # list all sessions"
echo "   tmux kill-session -t $SESSION   # kill"
