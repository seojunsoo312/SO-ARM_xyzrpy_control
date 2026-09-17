# YOLO — 1클래스 커스텀 OBB 검출

작업대 위 **한 종류** 물건을 찍고, **회전 박스(OBB)** 를 그린 뒤, YOLO11n-OBB를 파인튜닝해서 Orbbec 화면에서 잡는다.

COCO에 없는 물건이 맞다. 그래서 실시간 인식은 **우리가 학습한 `weights/best.pt`** 만 쓴다. `yolo11n-obb.pt` 는 학습 시작 가중치(전이학습)일 뿐이고, COCO 클래스로는 우리 물건을 찾지 않는다.

카메라는 `vision/camera.py` 와 같다. **640×480**, 기본 **180° 회전**. 펜던트·캘리브 스크립트와 **동시에 켜지 말 것**.

## 환경 (한 번)

YOLO·점군·펜던트는 **가상환경 하나**면 된다. 이름은 배포 때 바꿔도 되고, 코드는 환경 이름을 읽지 않는다. 지금 이 보드 실습은 `lerobot`. `setup.sh` 는 나중에 맞춘다.

시스템 `/bin/python` 은 쓰지 않는다 (비전 캘리브 전용).

```bash
# 프로젝트 루트에서
conda activate lerobot
```

학습 전에 펜던트·브라우저를 끄자. Orin Nano 메모리는 8GB다.

## 클래스 이름

기본 이름은 `yolo/cad/model.yaml` 의 `class` 다. 라벨 txt는 클래스 인덱스 **0** 이라, `prepare.py` 전에 이름을 바꿔도 된다.

## 순서

프로젝트 루트에서, `conda activate lerobot` 한 채로:

```bash
python yolo/train/capture.py     # SPACE=저장  q=종료
python yolo/train/label.py       # 꼭짓점 4클릭  rmb/d=취소  n=다음  p=이전  q=종료
python yolo/train/prepare.py     # train/val 목록만 (이미지 복사 없음)
python yolo/train/train.py       # YOLO11n-OBB. OOM 나면 --batch 2
python yolo/train/detect.py      # 실시간. q=종료
python yolo/pose/roi_cloud.py --mask   # 인스턴스 점군 → 최상단 (지금은 로컬 면 디버그)
```

라벨 한 줄은 `class x1 y1 x2 y2 x3 y3 x4 y4` (꼭짓점 4개, 0~1). 직각을 강제하지 않는다.

세그(마스크)가 필요하면 OBB 라벨 다음에:

```bash
python yolo/train/auto_seg.py      # OBB AABB → MobileSAM 마스크 → labels_seg
python yolo/train/review_seg.py    # 확인. d=마지막 삭제  r=비우기
python yolo/train/prepare.py --seg
python yolo/train/train.py --seg   # yolo11n-seg → weights/best-seg.pt
python yolo/train/detect.py --seg  # 마스크 + 마스크 중심
```

빈 책상은 점 없이 두면 마스크도 안 만든다. SAM이 틀린 장만 `review_seg`에서 지운다. 전 장 폴리곤을 손으로 칠 필요는 없다.

점군 ROI는 **OBB** 가 기본이다 (체크리스트). 세그 마스크로 책상 점을 더 빼고 싶으면:

```bash
python yolo/pose/roi_cloud.py --mask
python yolo/pose/roi_cloud.py --base    # T_base_cam 있으면 베이스 mm
```

`roi_cloud.py`는 V4L2가 아니라 Orbbec SDK다. Viewer·펜던트·`detect.py`와 같이 켜지 말 것. PLY는 `yolo/runs/roi/`.

`roi_cloud.py` 처리 순서:

```
YOLO OBB (쌓임은 --mask)
→ 전체 화면에서 RANSAC 책상 평면 추정
→ 검출마다 책상보다 위인 점군만 (합치지 않음)
→ 높이 점수로 최상단 1개
→ (현재) 윗면 슬라이스 → 로컬 면 PCA (디버그)
→ (예정) 그 ROI vs CAD → FPFH+RANSAC → Point-to-Plane ICP → xyzrpy
→ s 키로 선택 인스턴스 ply
```

6D 정본은 CAD 축 기준 \(xyzrpy\) 다. 설계는 `docs/grasp_pose_pipeline.md` 0절 수정안. CAD 는 `yolo/cad/` (`model.yaml` 의 `mesh`). 로컬 면은 등록 전까지 화면 확인용이다.

- `r`: 책상 평면 다시 추정
- `v`: 선택 인스턴스를 Open3D로 보기 (`pose/view/view_cloud.py`)
- `s`: 선택 인스턴스 ply 저장
- `--slice-mm 6`: 윗면 슬라이스 두께
- `--plane-mm 6`: 책상에서 6mm 이내 제거 (얇은 물체는 낮출 것)
- `--no-plane`: 비교용으로 책상 제거 끄기
- `--base`: `vision/calib_data/eye_to_hand.json` 의 `T_base_cam` 으로 베이스 mm

학습 `data.yaml` 의 `path` 는 `prepare.py`가 이 머신에서 `yolo/` 절대경로로 쓴다. Ultralytics가 홈의 `datasets_dir`에 상대경로를 붙이지 않게 하기 위함이다. train/val 목록은 `datasets/splits/`에 있다.

GPU·카메라만 먼저 보고 싶으면 (우리 물건은 안 잡힘):

```bash
python yolo/train/detect.py --pretrained
```

## 찍을 때

- 카메라·프레임은 고정. 물건만 움직인다.
- **80장 이상**. 위치, 회전, 거리, 조명, 손·그리퍼 가림, **빈 책상**(박스 없이 저장).
- 한 화면에 같은 물건이 여러 개면 OBB도 여러 개.

## 폴더

```
yolo/
├── README.md
├── setup.sh              # conda에 Jetson torch + ultralytics
├── requirements.txt
├── config.py             # 경로, batch, CAD yaml
├── data.yaml             # prepare가 씀 (path= yolo/ 절대경로)
├── train/                # 촬영 → 라벨 → 학습 → 검출
│   ├── boxes.py          # 픽셀 4점 ↔ YOLO OBB txt
│   ├── capture.py
│   ├── label.py          # 꼭짓점 4클릭
│   ├── auto_seg.py       # OBB AABB → MobileSAM 마스크
│   ├── review_seg.py
│   ├── prepare.py        # train/val 목록 8:2  (--seg 면 폴리곤)
│   ├── train.py          # YOLO11n-OBB / --seg 면 yolo11n-seg
│   └── detect.py         # 실시간. --seg 면 마스크 중심
├── pose/                 # 점군 · CAD 등록 (xyzrpy 정본)
│   ├── depth_cloud.py    # 뎁스 → 점군, 책상 RANSAC
│   ├── instances.py      # 인스턴스 점군 + 최상단
│   ├── register.py       # CAD vs ROI → FPFH+ICP
│   ├── roi_cloud.py      # 카메라 루프. 최상단 오버레이
│   ├── debug/            # 로컬 면 PCA (화면 확인·폴백)
│   │   └── local_plane.py
│   └── view/             # 본선 뒤 유틸
│       ├── view_cloud.py # Open3D 점군 뷰어
│       └── view_cad.py   # yolo/cad STL 파일 축(XYZ)
├── cad/                  # model.yaml + stl/ply (mm)
├── datasets/
│   ├── raw/
│   │   ├── images/
│   │   ├── labels/
│   │   └── labels_seg/   # 폴리곤. 박스 라벨은 그대로 둠
│   ├── splits/           # train.txt / val.txt (경로만)
│   └── seg/              # --seg 일 때 심볼릭 링크 (images→raw/images, labels→labels_seg)
├── weights/              # best.pt, best-seg.pt, mobile_sam.pt
└── runs/
    ├── train/
    ├── train_seg/
    └── roi/              # roi_cloud.py 가 저장한 ply
```

지금 안 하는 것: 집기(`pick.py`), 장면 전체 FPFH, YOLOv8-Pose+PnP, 펜던트에 YOLO 검출 UI.  
집기 티칭 GUI는 `pendant/teach_grasp.py` (`python pendant/main.py --grasp`).  
CAD 파일명은 `yolo/cad/model.yaml` 의 `mesh`.
