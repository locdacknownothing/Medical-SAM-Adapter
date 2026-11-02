#!/usr/bin/env bash
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <TASK_ID> [extra_train_flags...]"
    exit 1
fi

TASK_ID="$1"
shift

mkdir -p scratch/smoke
EXP="smoke_${TASK_ID}_$(date +%Y%m%d_%H%M%S)"
OUT_FILE="scratch/smoke/${EXP}.out"

# 1. Check GPU availability (wait up to 10 minutes if < 6000 MiB free)
WAIT_SECS=0
while true; do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [ -n "$FREE_MEM" ] && [ "$FREE_MEM" -ge 6000 ]; then
        echo "[smoke_test] GPU free memory: ${FREE_MEM} MiB (>= 6000 MiB). Proceeding."
        break
    fi
    if [ "$WAIT_SECS" -ge 600 ]; then
        echo "BLOCKED: GPU occupied (free memory: ${FREE_MEM:-unknown} MiB after 600s)"
        exit 2
    fi
    echo "[smoke_test] Waiting for GPU memory (currently ${FREE_MEM:-unknown} MiB free)..."
    sleep 30
    WAIT_SECS=$((WAIT_SECS + 30))
done

# 2. Run smoke test under flock to serialize GPU access
(
    flock -x 200
    echo "[smoke_test] Starting experiment: $EXP"
    PYTHONUNBUFFERED=1 SMOKE_MAX_EPOCHS=2 SMOKE_MAX_BATCHES=60 timeout --signal=INT --kill-after=30 600 \
        /workspace/miniconda3/envs/sam_adapt/bin/python train.py \
            -dataset DRIVE -net sam -mod sam_adpt \
            -sam_ckpt ./checkpoint/sam/sam_vit_h_4b8939.pth \
            -image_size 592 -out_size 592 -b 1 -data_path data/drive_aug/ \
            -vis 1 -val_freq 1 -encoder vit_h -loss_func dice_ce \
            -exp_name "$EXP" "$@" 2>&1 | tee "$OUT_FILE"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ] && [ "$RC" -ne 124 ] && [ "$RC" -ne 130 ]; then
        echo "[smoke_test] FAIL: train.py exited with unexpected code $RC"
        exit "$RC"
    fi
) 200>/tmp/msa_smoke_gpu.lock

# 3. Automated verification of output & logs
/workspace/miniconda3/envs/sam_adapt/bin/python - "$EXP" "$OUT_FILE" <<'EOF'
import glob
import math
import re
import sys

exp = sys.argv[1]
out_file = sys.argv[2]

with open(out_file, "r", errors="replace") as f:
    out_text = f.read()

log_files = sorted(glob.glob(f"logs/{exp}_*/Log/*.log"))
if not log_files:
    print(f"[smoke_test] FAIL: No log file found matching logs/{exp}_*/Log/*.log")
    sys.exit(1)

log_file = log_files[-1]
with open(log_file, "r", errors="replace") as f:
    log_text = f.read()

combined = out_text + "\n" + log_text

# Check 1: No Traceback or CUDA OOM
if "Traceback (most recent call last):" in combined:
    print("[smoke_test] FAIL: Traceback detected in output/log!")
    sys.exit(1)
if "CUDA out of memory" in combined or "OutOfMemoryError" in combined:
    print("[smoke_test] FAIL: CUDA out of memory detected!")
    sys.exit(1)

# Check 2: Trainable params summary
m_params = re.search(r"Trainable params summary: total_trainable=([\d,]+)", log_text)
if not m_params:
    print("[smoke_test] FAIL: 'Trainable params summary:' not found in log!")
    sys.exit(1)
trainable_count = int(m_params.group(1).replace(",", ""))
if trainable_count <= 0:
    print(f"[smoke_test] FAIL: total_trainable={trainable_count} <= 0!")
    sys.exit(1)

# Check 3: Train loss for epoch 0 and epoch 1
train_losses = {}
for m in re.finditer(r"Train loss:\s*([^\s|]+)\s*\|\|\s*@\s*epoch\s*(\d+)", log_text):
    val_str, ep_str = m.group(1), m.group(2)
    val = float(val_str)
    if math.isnan(val) or math.isinf(val):
        print(f"[smoke_test] FAIL: Non-finite train loss {val} at epoch {ep_str}!")
        sys.exit(1)
    train_losses[int(ep_str)] = val

if 0 not in train_losses or 1 not in train_losses:
    print(f"[smoke_test] FAIL: Expected train losses for epochs 0 and 1, got: {train_losses}")
    sys.exit(1)

ep0_loss = train_losses[0]
last_ep = max(train_losses.keys())
last_loss = train_losses[last_ep]
if last_loss > 1.05 * ep0_loss:
    print(f"[smoke_test] FAIL: Last train loss ({last_loss:.6f}) > 1.05 * epoch 0 loss ({ep0_loss:.6f})!")
    sys.exit(1)

# Check 4: Validation Total score & DICE
val_scores = []
for m in re.finditer(r"Total score:\s*([^\s,]+),\s*IOU:\s*([^\s,]+),\s*DICE:\s*([^\s|]+)\s*\|\|\s*@\s*epoch\s*(\d+)", log_text):
    tol, iou, dice, ep = float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
    if any(math.isnan(x) or math.isinf(x) for x in (tol, iou, dice)):
        print(f"[smoke_test] FAIL: Non-finite validation metrics at epoch {ep}!")
        sys.exit(1)
    val_scores.append((ep, tol, iou, dice))

if not val_scores:
    print("[smoke_test] FAIL: No validation 'Total score: ..., IOU: ..., DICE: ...' found in log!")
    sys.exit(1)

print("============================================================")
print(f"[smoke_test] PASS: {exp}")
print(f"  Log file         : {log_file}")
print(f"  Trainable params : {trainable_count:,}")
print(f"  Epoch 0 loss     : {ep0_loss:.6f}")
print(f"  Epoch {last_ep} loss     : {last_loss:.6f} (<= 1.05 * ep0 = {1.05 * ep0_loss:.6f})")
for ep, tol, iou, dice in val_scores:
    print(f"  Val @ epoch {ep}    : loss={tol:.6f}, IOU={iou:.6f}, DICE={dice:.6f}")
print("============================================================")
EOF
