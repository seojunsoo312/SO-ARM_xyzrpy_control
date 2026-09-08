#!/usr/bin/env bash
# Jetson Orin Nano Super / JetPack 6 (R36)
# 펜던트와 같은 conda 환경에 torch / ultralytics 를 넣는다.
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME=lerobot
CUDSS_URL="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-sbsa/libcudss-linux-sbsa-0.7.1.4_cuda12-archive.tar.xz"

CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
else
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    if [[ -f "${candidate}" ]]; then
      CONDA_SH="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
  echo "conda 가 없습니다. miniconda 경로를 확인하세요."
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_SH}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda 환경이 없습니다: ${ENV_NAME}"
  exit 1
fi
conda activate "${ENV_NAME}"

install_cudss() {
  if [[ -e "${CONDA_PREFIX}/lib/libcudss.so.0" ]]; then
    return 0
  fi
  echo "JetPack 기본 CUDA 에 cuDSS 가 없어서 conda 환경 lib 에 넣습니다."
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' RETURN
  curl -L --retry 3 -o "${work}/cudss.tar.xz" "$CUDSS_URL"
  tar -C "$work" -xf "${work}/cudss.tar.xz"
  find "$work" -name 'libcudss*.so*' -exec cp -a {} "${CONDA_PREFIX}/lib/" \;
}

python -m pip install -U pip wheel
python -m pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
install_cudss
python -m pip install -r requirements.txt
python - <<'PY'
import cv2
import numpy as np
import torch

print("python", __import__("sys").executable)
print("numpy", np.__version__)
print("cv2", cv2.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA 가 안 보입니다. Jetson 휠이 아닌 CPU torch 가 깔렸을 수 있습니다.")
print("gpu", torch.cuda.get_device_name(0))
from ultralytics import YOLO  # noqa: F401
print("ultralytics ok")
PY
echo "OK.  conda activate ${ENV_NAME}"
