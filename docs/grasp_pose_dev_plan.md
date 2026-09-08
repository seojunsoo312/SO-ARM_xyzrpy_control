# 집기 포즈 개발 계획

설계의 상세는 [`grasp_pose_pipeline.md`](./grasp_pose_pipeline.md) 를 따른다.  
이 문서는 **무엇을 어떤 순서로 구현·검증할지** 만 적는다.

구현 위치는 `yolo/`. 학습·캘리브·펜던트는 다시 만들지 않는다.

---

## 0. 수정안

정본 포즈를 **로컬 윗면 PCA** 에서 **CAD vs ROI 점군, FPFH+RANSAC → Point-to-Plane ICP** 로 바꾼다.

- \(xyzrpy\) 의 reference 는 CAD 축이다.
- FPFH 는 장면 전체가 아니라 **선택한 인스턴스 점군** 에만.
- 이미 있는 `grasp_pose.py` (인스턴스·최상단·로컬 면) 는 유지. 로컬 면은 디버그/폴백.
- 다음에 만들 것: `yolo/register.py`, `yolo/cad/` 모델, `roi_cloud` 등록 오버레이.

---

## 1. 현재 상태

### 되어 있는 것

- `K`, `T_base_cam` (`vision/calib_data/`)
- YOLO detect / seg
- 인스턴스 점군, 최상단, 로컬 면 오버레이 (`grasp_pose.py` + `roi_cloud.py`)
- 책상 RANSAC (바닥 제거)
- `2D_pose.ipynb`
- 펜던트 / `motion/`

### 없는 것

- CAD 파일 (`yolo/cad/`)
- `register.py` (FPFH+RANSAC, ICP, \(xyzrpy\))
- 클릭으로 인스턴스 고정
- `pick.py`

### 하지 않는 것

장면 전체 FPFH, DenseFusion, **YOLOv8-Pose + PnP (폐기)**, 펜던트에 검출 UI.

---

## 2. 원칙

1. **쌓임 가정.** 책상은 바닥 제거만.
2. 검증 장면: 1개 → 2–3개 → 여러 개(안 겹침) → 겹침 → 쌓임.
3. 더미에서 깨지면 포즈보다 **마스크·점 섞임** 먼저.
4. 로봇 명령의 정본은 **등록 \(T\)**. 로컬 면·2D 는 디버그.
5. 카메라와 시리얼은 집기 스크립트 외에는 한 프로세스만.

---

## 3. 마일스톤

| 단계 | 이름 | 산출 | 로봇 |
|------|------|------|------|
| B | 인스턴스 점군 | 물체마다 ply, 최상단 | 없음 (됨) |
| C0 | CAD 로드 | mm 점군, voxel·법선 | 없음 |
| C | 등록 6D | \(T\), \(xyzrpy\), fitness | 없음 |
| D | 디버그 비교 | 로컬 면 / 2D vs \(T\) | 없음 |
| E | 단층 집기 | `pick.py` | 있음 |
| F | 소량·겹침 | 선택·마스크 | 선택 |
| G | 쌓임 | 최상단 반복 | 있음 |

본선은 B → C0 → C. E 전에 C 를 화면에서 납득한다.

로컬 면 유틸(옛 C)은 이미 있다. 정본이 아니다.

---

## 4. 단계별 작업

### B. 인스턴스 점군 + 선택 (됨)

`python yolo/roi_cloud.py --mask`

통과: 1개·2–3개에서 개수와 최상단이 맞음. concatenate OBB 없음.

### C0. CAD → 모델 점군

**파일:** `yolo/cad/` (ply/mesh), `register.py` 로드 함수.

- 단위 mm. 장면 ROI 와 voxel 크기·법선 방향을 맞춤.
- 한 번 로드해서 재사용.

**완료:** `view_cloud` 로 CAD 점군과 저장 ROI ply 를 같은 스케일로 볼 수 있음.

### C. FPFH + RANSAC → ICP

**파일:** `yolo/register.py`. `roi_cloud.py` 가 선택 점군을 넘김.

1. 양쪽 voxel, 법선, FPFH.
2. RANSAC global registration → \(T_\text{init}\).
3. Point-to-Plane ICP → \(T\).
4. 카메라면 `apply_T` / `T_base_cam` 으로 베이스.
5. \(R,t\) → \(xyzrpy\). 화면에 축, fitness.

전 장면 금지.

**완료:** 책상 1개, 가운데에서 CAD가 ROI에 겹치고, 손으로 돌린 yaw와 \(rpy\) 가 맞음. 가장자리에서 뒤집히면 voxel/법선부터.

**검증 장면:** 1개 (가운데 → 가장자리). 통과 전 E 금지.

### D. 디버그 비교

로컬 면·`2D_pose.ipynb` 와 \(T\) 의 Δ. 로봇에는 등록만.

### E. 단층 집기

`pick.py`. 등록 \(T\) (또는 `T @ T_cad_grasp`). pre-grasp 후 하강. 펜던트 OFF.

### F / G. 소량 · 쌓임

선택·세그 검증. 섞인 ROI에 등록하지 않음. 최상단만 집고 루프.

---

## 5. 파일 작업 요약

| 파일 | B | C0 | C | D | E |
|------|---|----|---|---|---|
| `yolo/grasp_pose.py` | 됨 | | | 디버그 | |
| `yolo/register.py` | | 추가 | 추가 | | |
| `yolo/cad/` | | 추가 | | | |
| `yolo/roi_cloud.py` | 됨 | | 수정 | | |
| `yolo/pick.py` | | | | | 추가 |
| `yolo/README.md` | | 갱신 | 갱신 | | 갱신 |

---

## 6. 검증 체크리스트

| # | 장면 | 통과 기준 | 막는 단계 |
|---|------|-----------|-----------|
| 1 | 책상 1개, 가운데 | CAD↔ROI 겹침, \(xyzrpy\) 납득 | C |
| 2 | 책상 1개, 가장자리 | 뒤집힘 없음 | C |
| 3 | 떨어진 2–3개 | 최상단만 등록 | B, F |
| 4 | 1개 집기 | pre-grasp 후 집힘 | E |
| 5 | 안 겹친 여러 개 | 고른 것만 | F |
| 6 | 살짝 겹침 | 마스크가 갈라짐 | F (세그) |
| 7 | 작은 더미 | 위부터 2–3개 | G |

1–2 실패를 집기로 가리지 않는다.

---

## 7. 환경

- 점군·YOLO·Open3D 연산·집기: 펜던트와 **같은** 가상환경 하나. 이 보드 실습 이름은 `lerobot`. 배포 때 이름 바꿔도 됨.
- 렌즈·손-눈 재캘리브만 `/bin/python`
- 640×480, 컬러·뎁스 동일 180° 회전
- Open3D GUI: `/usr/bin/Open3D`. pip 뷰어 금지
- CAD 와 장면 점군 단위는 mm

---

## 8. 일정

```text
B (인스턴스, 됨)
 → C0 (CAD 로드)
 → C (FPFH+ICP, 화면) ─┬→ D (디버그, 병렬)
                        └→ E (단층 집기)
                             → F → G
```

- **1차:** C0+C+검증 1–2. 화면만.
- **2차:** E+검증 4.
- **3차:** F+G.

---

## 9. 리스크

| 증상 | 의심 (먼저) | 나중에 |
|------|-------------|--------|
| 등록이 180° 뒤집힘 | 법선 방향, 평면 CAD, voxel | `T_base_cam` |
| fitness 낮은데 \(T\) 사용 | ROI에 옆 물체·책상 | 세그 |
| 2개가 한 마스크 | 세그 데이터 | FPFH 파라미터 |
| 집을 때 바닥을 긁음 | TCP, pre-grasp, CAD 원점≠파지점 | |
| 더미에서 아래를 집음 | 선택 점수, 마스크 섞임 | ICP |
| mm/m 섞임 | CAD 단위 | voxel |

---

## 10. 다음 액션

1. CAD 파일(mm ply/mesh)을 `yolo/cad/` 에 둔다.
2. `register.py` 를 오프라인 ROI ply 로 돌린다.
3. `roi_cloud.py --mask` 에 \(xyzrpy\) 를 붙인다.
4. 검증 1–2 통과 전 `pick.py` 를 만들지 않는다.

`yolo/README.md` 는 검출·ROI 는 현재 코드, 6D 정본은 이 수정안과 맞춘다.
