# Jetson 설치

교육용 Jetson Orin Nano Developer Kit에 이 프로젝트를 까는 순서입니다. 한 대를 이 순서로 끝까지 맞춘 뒤(마스터), 그 디스크를 나머지 보드에 복제하는 것을 권합니다.

> 패키지 저장소와 노트북의 Python 3.10 시험 환경(torch 2.8, 이 fork)으로 확인한 순서입니다. Jetson 실물에서 처음 깔 때 각 단계 끝의 **확인** 명령을 꼭 돌려 보고, 다르게 나온 곳은 이 문서를 고칩니다.

> Jetson에 Claude Code가 있으면 `docs/jetson_claude_prompt.md`의 프롬프트로 이 문서의 확인을 한 번에 맡길 수 있습니다.

## 0. 먼저 알아 둘 것

- 환경은 **`AIvision` 하나, Python 3.10** 입니다. 펜던트, 노트북 실습, 카메라 창, YOLO 학습, 손눈 캘리브를 모두 여기서 실행합니다.
- 왜 3.10인가: Jetson용 GPU PyTorch(`pypi.jetson-ai-lab.io/jp6/cu126`)는 Python 3.10용만 있습니다. 7축 lerobot fork는 3.10에서 돌도록 고친 브랜치(`feat/so101-7dof-py310`)를 씁니다.
- **conda로는 Python만** 만듭니다. numpy 같은 패키지를 conda로 받으면 Jetson의 수학 라이브러리(NVPL)와 엮여 `libcblas.so.3: undefined symbol: nvpl_blas_core_...`로 numpy가 깨집니다. 나머지는 전부 pip로 깝니다.
- 수업에서는 터미널 두 개를 씁니다. 하나는 카메라 창, 하나는 펜던트나 노트북입니다. 둘 다 같은 `AIvision` 환경입니다.

```bash
# 터미널 1 (카메라)
conda activate AIvision && cd ~/AIvision && python yolo/pose/roi_cloud.py
# 터미널 2 (팔)
conda activate AIvision && cd ~/AIvision && python pendant/main.py --grasp
```

## 1. 준비물

- JetPack 6을 올린 Jetson (Ubuntu 22.04). 버전 확인: `cat /etc/nv_tegra_release` → `R36`
- 인터넷 연결 (설치하는 동안만)
- **ARM64용** Orbbec SDK v1.10 패키지. 노트북의 `..._linux_x64_release`는 Jetson에서 안 됩니다. [OrbbecSDK releases](https://github.com/orbbec/OrbbecSDK/releases)에서 `arm64`(또는 `aarch64`) 판을 받습니다.
- 노트북의 `yolo/weights/best.pt` (학습한 가중치. git에 없음. USB로 복사). 이것 하나만 옮깁니다.
  - 시작 가중치 `yolo11n.pt`는 옮기지 않습니다. 처음 학습할 때 ultralytics가 인터넷에서 받습니다.
  - 카메라 USB 권한 파일 `99-obsensor-libusb.rules`도 옮기지 않습니다. SDK 압축 안 `script/`에 들어 있습니다.

## 2. 보드 기본 설정

```bash
sudo apt update
sudo apt install -y git curl build-essential fonts-noto-cjk tk8.6 libtk8.6
sudo usermod -aG dialout,video $USER     # 시리얼(팔)과 카메라 권한. 다시 로그인해야 적용
grep POWER_MODEL /etc/nvpmodel.conf      # 모드 번호 보기. 이름이 MAXN 으로 시작하는 번호를 쓴다
sudo nvpmodel -m 2                       # Orin Nano Super 는 2 = MAXN_SUPER (0 은 15W 로 가장 낮다)
```

전원 모드 번호는 보드마다 다릅니다. Orin Nano Super는 `0=15W, 1=25W, 2=MAXN_SUPER`라서 `-m 0`을 주면 오히려 가장 느려집니다. `sudo nvpmodel -q`로 지금 모드를 봅니다.

메모리가 8GB라 YOLO 학습 때 모자랄 수 있습니다. 스왑을 8GB 만듭니다.

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

- `build-essential`이 없으면 lerobot 설치 중 `evdev` 빌드에서 멈춥니다(ARM64용 완성본이 없음).
- `fonts-noto-cjk`가 없으면 펜던트·카메라 창의 한글이 깨집니다.
- `tk8.6`, `libtk8.6`이 없으면 펜던트 한글이 네모로 나옵니다. 코드는 `/usr/lib/*-linux-gnu/libtk8.6.so`를 찾습니다.
- Chrome은 `chrome://settings/system`에서 「가능한 경우 그래픽 가속 사용」을 켜고 다시 시작합니다. 꺼져 있으면 WebGL이 없어 Meshcat 3D 화면이 흰색으로만 나옵니다. `chrome://gpu`의 WebGL이 「Hardware accelerated」이면 됩니다.

**확인**: `sudo nvpmodel -q`가 `MAXN`으로 시작, `free -h`에 Swap 8G 이상(기본 zram 약 3.7G + 스왑 파일 8G), `fc-list | grep -c "Noto Sans CJK"`가 0보다 큼.

## 3. 프로젝트와 카메라 SDK 놓기

프로젝트와 SDK를 **같은 폴더(홈)** 에 나란히 둡니다. 코드는 프로젝트 옆의 `OrbbecViewer_*` 또는 `OrbbecSDK_*` 폴더에서 `libOrbbecSDK.so`를 찾습니다(폴더 바로 아래나 `lib/` 아래).

```bash
cd ~
git clone https://github.com/seojunsoo312/SO-ARM_xyzrpy_control.git AIvision
# ARM64 SDK 압축을 홈에 풀기 → ~/OrbbecSDK_v1.10.xx_... 또는 ~/OrbbecViewer_v1.10.xx_...
mkdir -p ~/AIvision/yolo/weights && cp /media/$USER/<USB>/best.pt ~/AIvision/yolo/weights/

# 카메라 USB 권한: SDK 의 설치 스크립트가 규칙 복사와 다시 불러오기를 한다.
find ~/Orbbec* -name install_udev_rules.sh
sudo <위에서 나온 경로>
```

`find`가 아무것도 찍지 않으면 노트북의 `/etc/udev/rules.d/99-obsensor-libusb.rules`를 USB로 옮겨 같은 위치에 넣고 `sudo udevadm control --reload-rules && sudo udevadm trigger`를 합니다.

다른 위치에 SDK를 두면 `export ORBBEC_SDK_DIR=<그 폴더>`를 `~/.bashrc`에 적습니다.

**확인**: `find ~/Orbbec* -name "libOrbbecSDK.so*" | head -1`이 경로를 찍음. `ls /etc/udev/rules.d/99-obsensor-libusb.rules`가 있음. 카메라를 꽂고 `lsusb | grep 2bc5`가 두 줄.

## 4. conda 설치

이미 `~/miniconda3`나 `~/miniforge3`가 있으면 건너뜁니다. 없으면 ARM64용 Miniforge를 깝니다.

```bash
cd ~ && curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash && exec bash
```

## 5. `AIvision` 환경 (Python 3.10)

예전에 만든 `lerobot` 환경(3.12 등)이 있으면 지우고, `AIvision`으로 새로 만듭니다. 노트북 실습 커널 이름도 `AIvision`입니다.

순서가 중요합니다. **GPU torch를 먼저** 깔아야 lerobot과 ultralytics가 다른 torch를 받아 오지 않습니다.

```bash
conda deactivate
conda env remove -y -n lerobot          # 예전 lerobot 환경이 있을 때만
conda create -y -n AIvision python=3.10  # Python 만. 다른 패키지는 conda 로 받지 않는다.
conda activate AIvision

# 1) Jetson GPU torch. JetPack 6(CUDA 12.6)용 저장소. lerobot fork 는 torch>=2.7,<2.11.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 2) 7축 lerobot fork 의 Python 3.10 브랜치. 커밋을 고정한다.
pip install "lerobot[feetech] @ git+https://github.com/seojunsoo312/lerobot.git@f8e76c068ed4058e0504642ff9da8ea226032c54"

# 3) lerobot 이 opencv-python-headless(창 없는 판)를 깐다. 창이 필요하니 바꾼다.
#    4.12 부터는 numpy 2 를 요구하므로 4.11 로 둔다(5) 참고).
pip uninstall -y opencv-python-headless opencv-python
pip install -c <(printf 'torch==2.8.0\ntorchvision==0.23.0\n') "opencv-python<4.12"

# 4) 나머지. torch 를 다시 받지 않게 고정한다.
pip install -c <(printf 'torch==2.8.0\ntorchvision==0.23.0\n') ultralytics open3d pin meshcat customtkinter Pillow jupyter ipykernel
python -m ipykernel install --user --name AIvision --display-name AIvision

# 5) numpy 를 1.26 으로 내린다. 마지막에 한다.
pip install numpy==1.26.4
```

- 5)가 필요한 이유: Jetson 저장소의 torch 2.8.0 은 NumPy 1.x 로 빌드되어 있습니다. numpy 2 와 함께 쓰면 `torch.from_numpy`, `tensor.numpy()`가 `RuntimeError: Numpy is not available`로 실패하고 YOLO 추론이 멈춥니다. 같은 저장소의 torch 2.9.1, 2.10.0 은 `libcudss.so.0`이 없어 import 되지 않으므로 2.8.0 + numpy 1.26 으로 둡니다.
- 5) 뒤에 pip 가 `lerobot`, `cmeel-boost`, `rerun-sdk`가 numpy>=2 를 요구한다는 경고를 냅니다. 이 프로젝트가 쓰는 lerobot 모터 연결, pinocchio 기구학, open3d, YOLO 는 numpy 1.26.4 에서 확인했습니다. 무시합니다.

- 3)을 빼면 카메라 창에서 `cv2.imshow` 오류("The function is not implemented")가 납니다.
- `pip`가 opencv-python-headless가 없다는 경고를 내도 괜찮습니다.
- `<(printf ...)`는 bash 문법입니다. 제약 파일을 따로 만들지 않고 torch 버전을 고정합니다.

**확인**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.8.0 True
python -c "import torch, numpy as np; print(np.__version__, torch.from_numpy(np.ones(2)).cuda())"  # 1.26.4, 에러 없음
python -c "import lerobot.robots.so_follower as m, inspect; print('dof_mode' in inspect.signature(m.SO101FollowerConfig).parameters)"  # True
python -c "import cv2; cv2.namedWindow('t'); print('cv2 gui ok')"
python -c "import ultralytics, open3d, pinocchio, coal, meshcat, customtkinter; print('ok')"
cd ~/AIvision && python -c "from motion import Arm; a = Arm(); print(a.where()); a.close()"
```

마지막 줄은 브라우저에 가상 팔이 뜨면 성공입니다.

이렇게 나오면:

- `RuntimeError: Numpy is not available` 또는 `A module that was compiled using NumPy 1.x`: numpy 가 2.x 입니다. 5)를 다시 실행합니다.
- `cuda.is_available()`가 `False`이거나 torch 버전이 2.8.0이 아님: 2)~4)에서 다른 torch가 덮어썼습니다. 1)을 다시 실행합니다.
- `ImportError: libcusparseLt.so...`: Jetson용 cuSPARSELt를 깔아야 합니다. NVIDIA 포럼의 "PyTorch for JetPack 6" 글 순서를 따릅니다.
- `libcblas.so.3: undefined symbol: nvpl_blas_core_...`: numpy가 conda 쪽 라이브러리를 물었습니다. 환경을 지우고 5번을 처음부터 합니다. `conda install`은 쓰지 않습니다.

## 6. 팔 설정 (보드마다, 강사)

### 시리얼 포트 이름

코드는 팔로워를 `/dev/so101_follower`로 엽니다. 팔 보드의 시리얼 번호로 이름을 붙입니다. 팔을 하나씩 꽂고 번호를 읽습니다.

```bash
udevadm info -q property -n /dev/ttyACM0 | grep ID_SERIAL_SHORT
```

`/etc/udev/rules.d/99-serial.rules`에 그 보드 번호로 적습니다.

```text
SUBSYSTEM=="tty", ATTRS{serial}=="<팔로워 번호>", SYMLINK+="so101_follower"
SUBSYSTEM=="tty", ATTRS{serial}=="<리더 번호>", SYMLINK+="so101_leader"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/so101_*
```

### 리더·팔로워 캘리브

`AIvision` 환경에서 `lerobot-calibrate`로 합니다(7축, id는 `follower`, `leader`). 결과는 이 보드의 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json`, `.../teleoperators/so_leader/leader.json`에 저장되고, 펜던트 Connect가 그 파일을 읽습니다. 팔마다 다르므로 보드 사이에 복사하지 않습니다.

**확인**: `python pendant/main.py` → Connect → 팔이 튀지 않고 지금 자세에서 조그가 됨.

## 7. 수업 흐름 한 번 돌려 보기 (마스터에서)

모두 `AIvision` 환경, `~/AIvision`에서 실행합니다.

| 순서 | 명령 |
|---|---|
| 카메라 화면 | `python vision/handeye/vision_test.py` |
| 내부 값 | `python vision/handeye/get_intrinsic.py` |
| 손눈 촬영 | 펜던트 Connect 후 `python vision/handeye/capture.py` |
| 손눈 계산 | `python vision/handeye/compute.py` |
| 사진·라벨·학습 | `python yolo/train/capture.py` → `label.py` → `prepare.py` → `train.py` |
| 카메라 창 | `python yolo/pose/roi_cloud.py` |
| 물체 위치 불러오기 | 펜던트 `--grasp`의 버튼, 또는 노트북 `load_pick_targets(arm)` |

이때 같이 봅니다.

- `tegrastats`로 메모리. 카메라 창 + 펜던트 + 브라우저 + 노트북을 같이 켜도 스왑을 크게 쓰지 않는지.
- 카메라 창 첫 줄의 `device=0`(GPU). `cpu`면 5번 확인의 torch 줄을 다시 봅니다.
- 「물체 위치 불러오기」가 30초 안에 끝나는지. 넘으면 「카메라 창이 응답하지 않습니다」가 납니다(`yolo/pose/register_link.py`의 `REG_JOB_S`).
- 실행한 폴더에 `Log/`가 생기지 않는지.

## 8. 나머지 보드에 복제

마스터의 디스크를 이미지로 떠서 다른 보드에 씁니다. 한 대씩 설치하는 것보다 빠르고, 모든 보드가 같은 환경이 됩니다.

- microSD로 부팅: SD 카드 전체를 이미지로 떠서(예: `dd`, Balena Etcher) 같은 용량 이상의 카드에 씁니다.
- NVMe로 부팅: NVIDIA L4T의 백업·복원 도구(`l4t_backup_restore.sh`)를 씁니다.

복제한 뒤 보드마다 할 일입니다.

1. 컴퓨터 이름 바꾸기: `sudo hostnamectl set-hostname jetson-01`
2. 6번(시리얼 번호 규칙, 리더·팔로워 캘리브). 복제한 캘리브 파일은 마스터 팔의 값이라 그대로 쓰면 안 됩니다.
3. 카메라 캘리브와 YOLO 학습은 수업에서 학생이 합니다. 기수 사이에는 `python reset_class.py --all`. 이때 `yolo/weights/best.pt`도 지워지므로, 학생이 학습하기 전에는 카메라 창이 켜지지 않습니다.

## 9. 업데이트

코드를 고친 뒤 보드마다:

```bash
cd ~/AIvision && git pull
```

- lerobot fork를 바꾸면 `docs/requirements.txt`와 이 문서의 커밋 번호를 바꾸고, 보드마다 5번 2)를 새 번호로 다시 실행합니다.
- `yolo/weights/`는 git에 없으므로 바꿀 때는 USB로 복사합니다.
