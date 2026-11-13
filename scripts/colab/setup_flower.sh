#!/usr/bin/env bash
# FLOWER policy-server environment on Colab: own venv (torch), separate from the simulator venv.
#   bash scripts/colab/setup_flower.sh
set -euo pipefail
FLOWER_SHA=${FLOWER_SHA:-main}
[ -d /content/flower_vla_calvin/.git ] || git clone -q https://github.com/intuitive-robots/flower_vla_calvin.git /content/flower_vla_calvin
git -C /content/flower_vla_calvin checkout -q "$FLOWER_SHA"
echo "flower_vla_calvin @ $(git -C /content/flower_vla_calvin rev-parse HEAD)"
command -v uv >/dev/null || pip install -q uv
[ -x /content/flower-venv/bin/python ] || uv venv -q --python 3.10 /content/flower-venv
PY=/content/flower-venv/bin/python
uv pip install -q --python "$PY" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.2.2" "torchvision==0.17.2" "numpy<2" "transformers==4.44.2" timm einops einops_exts \
  "pytorch-lightning==2.0.8" "hydra-core==1.1.1" omegaconf safetensors huggingface_hub \
  sentence-transformers ftfy torchsde torchdiffeq termcolor "setuptools<70" \
  wandb plotly "moviepy<2" gitpython scipy matplotlib opencv-python-headless tqdm hydra-colorlog
mkdir -p /content/ckpt
"$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('mbreuss/flower_calvin_abc', local_dir='/content/ckpt/flower_calvin_abc'))"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
