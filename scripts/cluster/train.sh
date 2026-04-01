#!/bin/bash
#SBATCH --job-name=train-pcb-qwen
#SBATCH --partition=cluster02
#SBATCH --gpus=pro6000:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/train_pcb_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/train_pcb_%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit

export HF_HOME=/projects/_ssd/xrssd/cache/huggingface
export TMPDIR=/projects/_ssd/xrssd/tmp
export PYTHONUSERBASE=/projects/_ssd/xrssd/python_user

cd /projects/_ssd/xrssd/qwen-image-finetune

echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== Starting Training ==="
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    -m qflux.main \
    --config configs/pcb_harmonize_qwen_edit_2511.yaml

echo "=== DONE ==="
ls -la /projects/_ssd/xrssd/runs/pcb_harmonize_qwen_edit_v1/checkpoints/ 2>/dev/null
