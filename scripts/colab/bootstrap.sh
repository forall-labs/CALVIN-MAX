#!/usr/bin/env bash
# Fresh Colab runtime -> simulator venv + FLOWER venv + FLOWER server on :6100.
#   On a fresh Colab or Linux GPU runtime with this repo at /content/CALVIN-MAX:  bash scripts/colab/bootstrap.sh
set -euo pipefail
bash /content/CALVIN-MAX/scripts/colab/setup_calvin.sh
bash /content/CALVIN-MAX/scripts/colab/setup_flower.sh
pkill -f flower_server.py || true
cd /content && nohup /content/flower-venv/bin/python /content/CALVIN-MAX/scripts/adapters/flower_server.py \
  --ckpt /content/ckpt/flower_calvin_abc/model.safetensors --port 6100 > /content/flower_server.log 2>&1 &
for i in $(seq 1 60); do grep -q "FLOWER ready" /content/flower_server.log && break; sleep 5; done
grep "FLOWER ready" /content/flower_server.log
echo BOOTSTRAP_OK
