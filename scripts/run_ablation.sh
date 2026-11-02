#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/workspace/miniconda3/envs/sam_adapt/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DATASET="${DATASET:-DRIVE}"
DATA_PATH="${DATA_PATH:-data/drive_aug/}"
SAM_CKPT="${SAM_CKPT:-./checkpoint/sam/sam_vit_h_4b8939.pth}"
IMAGE_SIZE="${IMAGE_SIZE:-592}"
VAL_FREQ="${VAL_FREQ:-1}"
VIS="${VIS:-10}"
WARM="${WARM:-3}"
PATIENCE="${PATIENCE:-50}"

if [ $# -eq 0 ]; then
    TARGETS=("A0" "A1" "A2" "A3" "A4")
else
    TARGETS=("$@")
fi

run_single_ablation() {
    local target="$1"
    local exp_name=""
    local extra_flags=()

    case "$target" in
        A0|ablation_A0_conv_fixed)
            exp_name="ablation_A0_conv_fixed"
            extra_flags=(-adapter_type conv)
            ;;
        A1|ablation_A1_tubular)
            exp_name="ablation_A1_tubular"
            extra_flags=(-adapter_type tubular)
            ;;
        A2|ablation_A2_multilayer_vit)
            exp_name="ablation_A2_multilayer_vit"
            extra_flags=(-adapter_type tubular -multilayer_vit)
            ;;
        A3|ablation_A3_hq_decoder)
            exp_name="ablation_A3_hq_decoder"
            extra_flags=(-adapter_type tubular -multilayer_vit -hq_decoder)
            ;;
        A4|ablation_A4_full)
            exp_name="ablation_A4_full"
            extra_flags=(-adapter_type tubular -multilayer_vit -hq_decoder -hybrid_prompt)
            ;;
        *)
            echo "[run_ablation] ERROR: Unknown ablation target '$target'. Valid targets: A0 A1 A2 A3 A4"
            exit 1
            ;;
    esac

    echo "=================================================================="
    echo "[run_ablation] Starting Training: ${exp_name} (target=${target}, batch_size=${BATCH_SIZE})"
    echo "[run_ablation] Extra flags: ${extra_flags[*]}"
    echo "=================================================================="

    PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py \
        -dataset "$DATASET" -net sam -mod sam_adpt \
        -sam_ckpt "$SAM_CKPT" \
        -image_size "$IMAGE_SIZE" -out_size "$IMAGE_SIZE" \
        -b "$BATCH_SIZE" -data_path "$DATA_PATH" \
        -vis "$VIS" -val_freq "$VAL_FREQ" -warm "$WARM" -patience "$PATIENCE" \
        -encoder vit_h -loss_func dice_ce \
        -exp_name "$exp_name" "${extra_flags[@]}"

    local latest_dir
    latest_dir=$(ls -dt "logs/${exp_name}_"* 2>/dev/null | head -n 1 || true)
    if [ -z "$latest_dir" ]; then
        echo "[run_ablation] ERROR: Could not locate log directory for ${exp_name}"
        exit 1
    fi

    local ckpt="${latest_dir}/Model/best_dice_checkpoint.pth"
    if [ ! -f "$ckpt" ]; then
        echo "[run_ablation] ERROR: Checkpoint not found at ${ckpt}"
        exit 1
    fi

    echo "------------------------------------------------------------------"
    echo "[run_ablation] Running Final TTA Evaluation on ${ckpt}"
    echo "------------------------------------------------------------------"

    PYTHONUNBUFFERED=1 "$PYTHON_BIN" val.py \
        -dataset "$DATASET" -net sam -mod sam_adpt \
        -sam_ckpt "$SAM_CKPT" \
        -weights "$ckpt" \
        -image_size "$IMAGE_SIZE" -out_size "$IMAGE_SIZE" \
        -b "$BATCH_SIZE" -data_path "$DATA_PATH" \
        -vis 1 -encoder vit_h -loss_func dice_ce \
        -tta \
        -exp_name "val_tta_${exp_name}" "${extra_flags[@]}"

    echo "[run_ablation] Completed ${exp_name} (train + TTA eval)."
}

for t in "${TARGETS[@]}"; do
    run_single_ablation "$t"
done

echo "=================================================================="
echo "[run_ablation] All requested ablation runs (${TARGETS[*]}) finished!"
echo "=================================================================="
