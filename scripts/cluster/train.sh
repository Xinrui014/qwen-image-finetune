#!/bin/bash
#SBATCH --job-name=train-pcb-qwen
#SBATCH --partition=cluster02
#SBATCH --gpus=pro6000:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/train_pcb_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/train_pcb_%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit

export HF_HOME=/projects/_ssd/xrssd/cache/huggingface
export TMPDIR=/projects/_ssd/xrssd/tmp
export PYTHONUSERBASE=/projects/_ssd/xrssd/python_user
export PIP_CACHE_DIR=/projects/_ssd/xrssd/cache/pip
export CONDA_PKGS_DIRS=/projects/_ssd/xrssd/cache/conda_pkgs
export WANDB_API_KEY=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
export WANDB_DIR=/projects/_ssd/xrssd/runs

cd /projects/_ssd/xrssd/qwen-image-finetune

# Patch flash_attention_2 -> sdpa for Blackwell
sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' src/qflux/models/load_model.py

# Verify cache exists
CACHE_DIR=/projects/_ssd/xrssd/runs/pcb_harmonize_qwen_edit_v1/cache
CACHE_COUNT=$(find ${CACHE_DIR} -type f \( -name "*.pt" -o -name "*.safetensors" \) 2>/dev/null | wc -l)
echo "=== Cache check: ${CACHE_COUNT} files in ${CACHE_DIR} ==="
if [ "${CACHE_COUNT}" -lt 1000 ]; then
    echo "ERROR: Cache looks incomplete (${CACHE_COUNT} files). Aborting."
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== Starting Training (r=32, bs=4/gpu, 4x pro6000, 10k steps) ==="
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    -m qflux.main \
    --config configs/pcb_harmonize_qwen_edit_2511.yaml

echo "=== DONE ==="
find /projects/_ssd/xrssd/qwen-image-finetune/runs/pcb_harmonize_qwen_edit_v1/ -name "*.safetensors" -ls 2>/dev/null
