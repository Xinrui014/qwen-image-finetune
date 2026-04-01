#!/bin/bash
#SBATCH --job-name=setup-qwen-edit
#SBATCH --partition=cluster02
#SBATCH --nodelist=cpu-1
#SBATCH --time=2:00:00
#SBATCH --output=/projects/xrssd/logs/setup_qwen_edit_%j.out
#SBATCH --error=/projects/xrssd/logs/setup_qwen_edit_%j.err

set -e
WORK_DIR=~/xrssd

# ── 1. Clone qwen-image-finetune ──────────────────────────────────────────────
echo "=== Cloning qwen-image-finetune ==="
cd $WORK_DIR
if [ ! -d "qwen-image-finetune" ]; then
    git clone https://github.com/Xinrui014/qwen-image-finetune.git
else
    echo "Already cloned, pulling latest..."
    cd qwen-image-finetune && git pull && cd ..
fi

# ── 2. Create conda env ───────────────────────────────────────────────────────
echo "=== Setting up conda env ==="
module load Miniforge3
source activate

# Create env if not exists
if ! conda env list | grep -q "qwen_edit"; then
    echo "Creating qwen_edit env..."
    conda create -n qwen_edit python=3.12 -y
else
    echo "qwen_edit env already exists"
fi

conda activate qwen_edit

# ── 3. Install PyTorch + CUDA ─────────────────────────────────────────────────
echo "=== Installing PyTorch ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── 4. Install diffusers (latest from git for Qwen-Image-Edit support) ────────
echo "=== Installing diffusers + deps ==="
pip install "git+https://github.com/huggingface/diffusers"
pip install transformers accelerate sentencepiece protobuf
pip install peft safetensors

# ── 5. Install qwen-image-finetune deps ───────────────────────────────────────
echo "=== Installing qwen-image-finetune deps ==="
cd $WORK_DIR/qwen-image-finetune
pip install -r requirements.txt 2>/dev/null || echo "No requirements.txt or some deps failed"
pip install -e . 2>/dev/null || echo "No setup.py/pyproject.toml editable install"

# Additional deps
pip install opencv-python pillow tqdm numpy pandas pyyaml wandb

# ── 6. Download Qwen-Image-Edit-2511 model ────────────────────────────────────
echo "=== Downloading Qwen-Image-Edit-2511 ==="
export HF_HOME=$WORK_DIR/cache/huggingface
mkdir -p $HF_HOME

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen-Image-Edit-2511',
    cache_dir='$HF_HOME/hub',
    ignore_patterns=['*.md', '*.txt'],
)
print('Model downloaded successfully')
"

# ── 7. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "=== SETUP COMPLETE ==="
python -c "
import torch, diffusers, transformers, peft
print('PyTorch:', torch.__version__)
print('diffusers:', diffusers.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('CUDA available:', torch.cuda.is_available())
"
echo "HF cache: $(du -sh $HF_HOME 2>/dev/null)"
echo "qwen-image-finetune: $(ls $WORK_DIR/qwen-image-finetune/src 2>/dev/null)"
