#!/usr/bin/env bash
# Set up the CALVIN simulator on a Colab runtime, pinned to the commits in docs/CALVIN_NOTES.md.
# Uses an isolated Python 3.10 venv (uv-managed), so Colab's system Python version doesn't matter.
#
#   bash /content/CALVIN-MAX/scripts/colab/setup_calvin.sh
set -euo pipefail

ROOT=${ROOT:-/content/CALVIN-MAX}
TP=$ROOT/third_party
CALVIN_ENV_SHA=797142c588c21e76717268b7b430958dbd13bf48
CALVIN_SHA=fa03f01f19c65920e18cf37398a9ce859274af76

mkdir -p "$TP"
if [ ! -d "$TP/calvin_env/.git" ]; then
  git clone -q https://github.com/mees/calvin_env.git "$TP/calvin_env"
fi
git -C "$TP/calvin_env" checkout -q "$CALVIN_ENV_SHA"
git -C "$TP/calvin_env" submodule update -q --init --recursive
if [ ! -d "$TP/calvin/.git" ]; then
  git clone -q https://github.com/mees/calvin.git "$TP/calvin"
fi
git -C "$TP/calvin" checkout -q "$CALVIN_SHA"

command -v uv >/dev/null || pip install -q uv
[ -x "$ROOT/.venv/bin/python" ] || uv venv -q --python 3.10 "$ROOT/.venv"
PY=$ROOT/.venv/bin/python
# gym 0.21 has metadata that uv and pip>=24.1 reject, and a setup.py that setuptools>=66 rejects.
# Install it alone with an old pip/setuptools inside the venv; everything else goes through uv.
uv pip install -q --python "$PY" "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"
"$PY" -m pip install -q --no-build-isolation "gym==0.21.0"
uv pip install -q --python "$PY" \
  "numpy<1.24" pybullet hydra-core hydra-colorlog omegaconf numpy-quaternion \
  scipy opencv-python-headless cloudpickle gitpython pandas matplotlib numba
uv pip install -q --python "$PY" --no-deps -e "$TP/calvin_env"
uv pip install -q --python "$PY" -e "$ROOT"

# EGL for headless GPU rendering; Colab images normally ship libEGL via the NVIDIA driver.
"$PY" - <<'EOF'
import pybullet, pkgutil
print("pybullet", pybullet.__file__)
print("eglRenderer plugin found:", pkgutil.get_loader("eglRenderer") is not None)
EOF
echo "setup done: $PY"
