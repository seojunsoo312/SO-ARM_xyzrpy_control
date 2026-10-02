# 노트북 설치

개발용 노트북(x86, Ubuntu)에 이 프로젝트를 까는 순서입니다. Jetson과 **같은 Python 3.10, 같은 torch 2.8, 같은 lerobot fork 커밋**을 써서, 노트북에서 된 것이 Jetson에서도 되게 맞춥니다. Jetson은 `docs/jetson_setup.md`입니다.

## Jetson과 다른 것

| 항목 | 노트북 | Jetson |
|---|---|---|
| torch 저장소 | `download.pytorch.org/whl/cu128` (일반 NVIDIA GPU) | `pypi.jetson-ai-lab.io/jp6/cu126` (Jetson GPU) |
| Orbbec SDK | `..._linux_x64_release` | ARM64판 |
| 나머지 | 같음 | 같음 |

torch는 기기마다 다른 판입니다. 노트북 판을 Jetson에 깔면 GPU를 못 씁니다.

## 1. 시스템 패키지

```bash
sudo apt install -y git build-essential fonts-noto-cjk tk8.6 libtk8.6
sudo usermod -aG dialout,video $USER
```

카메라 USB 권한(`/etc/udev/rules.d/99-obsensor-libusb.rules`, SDK의 `script/install_udev_rules.sh`로 설치)과 팔 시리얼 이름(`99-serial.rules`)은 `docs/jetson_setup.md` 3번, 6번과 같습니다.

## 2. 카메라 SDK

x64 Orbbec SDK(`OrbbecViewer_v1.10.xx_..._linux_x64_release`)를 프로젝트 폴더와 같은 곳(홈)에 풉니다. 코드는 프로젝트 옆에서 찾습니다(`docs/vision.md`).

## 3. `AIvision` 환경 (Python 3.10)

conda로는 Python만 만들고 나머지는 pip로 깝니다. **torch를 먼저** 깝니다.

```bash
conda create -y -n AIvision python=3.10
conda activate AIvision

# 1) 노트북 GPU torch. Jetson 과 같은 2.8.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 2) 7축 lerobot fork (Jetson 과 같은 커밋)
pip install "lerobot[feetech] @ git+https://github.com/seojunsoo312/lerobot.git@f8e76c068ed4058e0504642ff9da8ea226032c54"

# 3) 창 있는 OpenCV 로 바꾸기
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python

# 4) 나머지
pip install ultralytics open3d pin meshcat customtkinter Pillow jupyter ipykernel
python -m ipykernel install --user --name AIvision --display-name AIvision

# 5) lerobot 이 torch 2.10 용 torchcodec 0.10 을 깐다. torch 2.8 에 맞는 0.7 로 바꾼다.
#    (Jetson 에는 torchcodec 이 깔리지 않으므로 이 줄이 없다)
pip install "torchcodec==0.7.*"
```

ROS 등으로 `PYTHONPATH`가 잡혀 있으면 다른 Python 패키지가 섞일 수 있습니다. 이 환경을 쓸 때는 `unset PYTHONPATH`를 먼저 합니다.

**확인**: `docs/jetson_setup.md` 5번의 확인 명령과 같습니다. `torch.cuda.is_available()`가 `True`면 됩니다.

## fork 코드를 고치면서 쓸 때

위 2)는 GitHub의 고정 커밋을 받습니다. `~/lerobot`을 직접 고치며 시험하려면 그 폴더를 `feat/so101-7dof-py310` 브랜치로 두고 `pip install -e ~/lerobot`으로 바꿉니다. 다 고친 뒤에는 커밋·push하고, 두 설치 문서와 `docs/requirements.txt`의 커밋 번호를 바꿉니다.
