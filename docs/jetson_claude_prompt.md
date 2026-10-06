# Jetson 점검을 Claude Code에 맡기기

Jetson에 Claude Code를 깔았으면, `~/AIvision`에서 `claude`를 실행하고 아래 블록을 그대로 붙여 넣습니다. 설치 순서는 `docs/jetson_setup.md`이고, 이 프롬프트는 그 문서의 확인 명령을 모두 돌려 통과/실패 표를 만들게 합니다.

붙여 넣기 전에 카메라를 꽂아 둡니다. 팔도 연결해 두면 시리얼 번호 확인까지 이어집니다. "지금까지 한 것"은 실제로 한 데까지로 고쳐서 붙입니다.

```text
~/AIvision 은 SO-101 7축 로봇팔 교육 프로젝트이고, 설치 순서는 docs/jetson_setup.md 에 있다.
이 Jetson 을 다른 보드에 복제할 마스터로 만드는 게 목표다.

지금까지 한 것:
- git clone (폴더 이름 ~/AIvision), git pull 로 최신까지 받음
- conda 환경 AIvision (Python 3.10) 설치. docs/jetson_setup.md 5단계대로 pip 로 설치함
- ~/AIvision/yolo/weights/best.pt 복사함

할 일:

1. Orbbec 카메라 SDK 설치 (docs/jetson_setup.md 3단계)
   - 노트북은 OrbbecViewer v1.10.27 x64판을 쓴다. Jetson 에는 같은 버전의 ARM64판을 홈(~)에 받아서 푼다:
     https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecViewer_v1.10.27_202509260950_arm64_release.zip
   - 코드는 프로젝트 폴더 옆(홈 바로 아래)의 OrbbecViewer_* 또는 OrbbecSDK_* 폴더에서,
     그 폴더 바로 아래나 lib/ 아래의 libOrbbecSDK.so 를 찾는다. 압축이 폴더를 한 겹 더 만들면 정리해라.
   - Viewer 압축에 libOrbbecSDK.so 가 없으면 같은 버전의 SDK판을 대신 받아라:
     https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C%2B%2B_v1.10.27_20250925_0549823_linux_arm64_release.zip
   - 카메라 USB 권한: SDK 안의 script/install_udev_rules.sh 를 sudo 로 실행한다.
     없으면 나에게 말해라 (노트북의 99-obsensor-libusb.rules 를 USB 로 가져오겠다).
   - 확인: find ~/Orbbec* -name "libOrbbecSDK.so*" 가 경로를 찍고,
     /etc/udev/rules.d/99-obsensor-libusb.rules 가 있고, 카메라를 꽂으면 lsusb | grep 2bc5 가 두 줄.

2. docs/jetson_setup.md 를 처음부터 읽고, 각 단계 끝의 "확인" 명령을 모두 실행해서
   단계별로 통과/실패를 표로 보여줘. 특히 아래를 꼭 확인해:
   - cat /etc/nv_tegra_release 가 R36 인지
   - torch 2.8.0, torch.cuda.is_available() == True
   - numpy 가 깨지지 않는지 (libcblas / nvpl 관련 undefined symbol 오류가 없는지)
   - lerobot 의 SO101FollowerConfig 에 dof_mode 가 있는지 (7축 fork 커밋 f8e76c06)
   - opencv 가 headless 판이 아니라 opencv-python 인지 (cv2.imshow 가 되는지)
   - 각 진입점이 --help 로 실행되는지: pendant/main.py, pendant/teach_grasp.py,
     yolo/pose/roi_cloud.py, yolo/pose/register.py, yolo/train/train.py, yolo/train/detect.py,
     vision/handeye/capture.py, vision/handeye/compute.py, reset_class.py
   - 가상 팔: cd ~/AIvision && python -c "from motion import Arm; a=Arm(open_browser=False); a.initial(); a.moveL([20,0,0],'rel'); print(a.where()); a.close()"
   - 카메라 (카메라를 쓰는 다른 프로그램은 끈 상태에서):
     cd ~/AIvision && python -c "
import sys; sys.path[:0]=['.', 'pendant']
from vision.rgbd import OrbbecV1
cam=OrbbecV1(); fr=[cam.grab() for _ in range(15)]; cam.close()
ok=[f for f in fr if f[0] is not None]; print('frames', len(ok), ok[-1][0].shape, ok[-1][1].shape)
from ultralytics import YOLO
r=YOLO('yolo/weights/best.pt').predict(ok[-1][0], device=0, verbose=False); print('yolo', r[0].boxes.data.device)"
     → 컬러 (480,640,3), 뎁스 (480,640), yolo 가 cuda:0 이면 정상.
     실행한 폴더에 Log/ 폴더가 새로 생기지 않아야 한다.
   - 펜던트 한글 글꼴: cd ~/AIvision/pendant && python -c "import ui_style; print(ui_style.enable_xft_tk())" → True

3. 실패한 항목은 원인을 찾아서 고쳐줘. 단, 지켜야 할 규칙:
   - conda 로는 Python 만 설치한다. numpy 등 패키지를 conda 로 깔지 마라 (Jetson 에서 numpy 가 깨진다).
   - torch 는 pypi.jetson-ai-lab.io/jp6/cu126 의 2.8.0 을 유지한다.
     다른 패키지를 설치하다가 torch 가 바뀌지 않게 해라 (설치 전후로 torch 버전 확인).
   - sudo 가 필요한 명령은 실행하기 전에 나에게 보여주고 물어봐.

4. 아직 안 한 6단계(팔 시리얼 이름 규칙 99-serial.rules, 리더·팔로워 캘리브)는
   내가 직접 해야 하는 일(팔 연결, 캘리브 동작)과 네가 할 수 있는 일을 나눠서 알려줘.
   99-serial.rules 는 팔마다 시리얼 번호가 달라서 이 보드에서 새로 만든다. 노트북 것을 복사하지 않는다.

5. 문서와 실제가 다른 곳(틀린 명령, 다른 순서, 빠진 패키지, ARM64 SDK 폴더 구성 등)을 찾으면
   docs/jetson_setup.md 를 고치고 무엇을 고쳤는지 알려줘. 커밋과 푸시는 내가 확인한 뒤에 한다.
```
