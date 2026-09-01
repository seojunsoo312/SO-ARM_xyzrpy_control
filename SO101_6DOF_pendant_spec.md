# SO101_6DOF 로봇팔 간이 티칭 펜던트 개발 명세서

## 1. 프로젝트 개요

본 프로젝트는 6자유도(6-DOF) 로봇팔인 SO101_6DOF (SO-101 기반, `elbow_roll` 모터 추가)을 제어하기 위한 GUI 기반의 독립형 티칭 펜던트 소프트웨어를 개발하는 것이다. 무거운 ROS 2 생태계를 배제하고, 순수 파이썬 환경에서 조그(Jog) 제어와 디지털 트윈(가상 시뮬레이션)을 구현한다.

- 팔 관절 6축 + 그리퍼 1축 (서보 총 7개). 카르테시안 IK는 팔 6축만 사용하고, 그리퍼는 독립 축이다.
- UI / 기구학 / 하드웨어 / **뷰어**를 모듈로 분리한다. GUI는 속도·목표 포즈 명령만 넣고, 제어 루프는 관절각 `q`와 끝단 상태만 돌려준다. IK는 뷰어 안에 두지 않는다.
- 가상 모드에서는 로봇 없이 3D 뷰만 동작해야 한다. 실물 모드는 기존 LeRobot Feetech 버스를 감싸서 사용하며, Feetech 시리얼 프로토콜을 새로 구현하지 않는다.
- **3D의 역할은 티칭 입력이 아니라 XYZ/RPY 디버그다.** 조그·go-to 후 TCP가 시킨 방향으로 갔는지, FK와 GUI 숫자가 맞는지 눈으로 확인한다. **첫 구현에 Meshcat 3D를 넣는다** (별도 브라우저 창). GUI와 한 창으로 합칠지, 기즈모로 집을지는 **후순위**이지, 3D 자체를 미루는 것이 아니다. 계약은 `Visualizer.display(q)`만 고정한다.

참고 구현(동작 검증용, 구조는 답습하지 말 것): `examples/so101_ee_gui/so101_ee_gui.py`. 이 파일은 UI·IK·모터가 한 클래스에 붙어 있다. 본 펜던트는 그 제어 튜닝(DLS 서보, 홀드 축, 스무딩, 캘리브 오프셋)만 가져오고 백엔드를 분리한다.

## 2. 핵심 기술 스택 및 필수 라이브러리

ROS 2 없이 순수 Python으로 구동한다. 에이전트는 아래 패키지를 기반으로 작성한다.

* **수학 및 역기구학 엔진:** Pinocchio (`pin` 패키지)
  - 역할: URDF 파싱, FK, 자코비안. IK는 Pinocchio 자코비안 위의 DLS 서보로 구현한다. 원샷 \(J^{\dagger}\) / 원샷 LM에 의존하지 않는다. **뷰어에서 IK를 돌리지 않는다.**
* **GUI 프레임워크:** `customtkinter`
  - 역할: 펜던트 창, 조그 버튼, 좌표/관절 입력, 상태 표시. 3D를 이 창에 임베드하지 않는다 (후순위).
* **3D 시각화 (디버그):** Pinocchio `MeshcatVisualizer`
  - 역할: `q`를 브라우저에 그려 XYZ/RPY가 맞는지 확인. 기즈모·ghost·한 창 통합은 하지 않는다.
* **수치 연산:** `numpy`
* **하드웨어:** 기존 LeRobot `SOFollower` / `FeetechMotorsBus` 래핑
  - Feetech STS3215 UART 프로토콜·캘리브 JSON·모터 ID 맵을 재구현하지 않는다. `pyserial`은 그 스택의 의존성으로만 존재한다.
* **동시성:** `threading`
  - UI가 멈추지 않도록 IK·통신 루프를 백그라운드로 분리

**[환경 구축]**

Pinocchio는 PyPI 이름이 `pinocchio`가 아니다. 아래 중 프로젝트 환경에 맞는 것을 사용한다.

```text
pip install pin customtkinter meshcat numpy
# 또는 conda-forge: conda install -c conda-forge pinocchio
```

LeRobot 워크스페이스에서 실행하는 경우 `uv run` / 기존 `lerobot` 환경을 우선한다. 실물 모드는 `lerobot`의 Feetech extra가 필요하다.

## 3. 전체 시스템 아키텍처

### 3.1 스레드

* **UI 스레드 (Main):** CustomTkinter 렌더링, 버튼/입력 이벤트를 명령 객체로만 기록. IK·시리얼·FK를 UI 콜백에서 직접 호출하지 않는다.
* **제어 스레드 (Background):** 목표 **30 Hz**, 가능하면 50 Hz. 한 주기 안에 (상태 읽기) → IK/적분 → 안전 가드 → 뷰어에 `q` 게시 → (실물이면 모터 write). 주기를 못 지키면 해당 틱을 스킵하고, 밀린 명령을 한 번에 몰아서 보내지 않는다. 뷰어 푸시는 §3.4처럼 **최신 `q`만 30 Hz** 로 합친다 (제어 루프가 더 빨라도 JS/ZMQ를 틱마다 호출하지 않음).

### 3.2 모듈 경계 (필수)

```text
GUI  ──cmd──►  Controller  ──q──►  Visualizer (디버그, 별도 창 OK)
                    │
                    └──q──►  Hardware (Real 모드만)
```

* `gui_app.py`: 위젯과 이벤트. 로봇/Pinocchio를 import하지 않는 것을 목표로 한다. 불가피하면 얇은 콜백만.
* `controller.py`: 조그·go-to·안전·모드 전환의 단일 루프. `q`의 소유자.
* `robot_kinematics.py`: URDF, FK, 자코비안, DLS IK, TCP 오프셋. 모터/GUI/뷰어 모름.
* `hw_controller.py`: `SOFollower` 래퍼. 엔코더 읽기/목표각 쓰기, 토크 on/off, 캘리브 오프셋. IK 모름.
* `visualizer.py`: `display(q)` 퍼사드. 이번 버전 구현은 Meshcat. IK·시리얼·사용자 입력 없음.

명령(`cmd`)과 상태(`state`) 계약은 대략 다음과 같다.

* `cmd`: `mode`, Cartesian twist 또는 joint velocity, go-to 목표(포즈 또는 관절), gripper, stop/home/estop
* `state`: `q` (URDF deg, 팔 6 + gripper), `ee` (TCP mm + RPY deg), `q_meas`(실물일 때), `ee_err_mm`, `mode`, `fault`

### 3.3 `q` 소유권

* 제어 루프가 **명령 관절각 `q_cmd`** 를 적분/서보한다.
* Real 모드: 매 주기 엔코더 `q_meas`를 읽고, IK 시드는 `(1-α) q_cmd + α q_meas`처럼 측정값을 약하게 섞는다. 매 주기 `q_meas`로 완전히 덮어쓰지 않는다(떨림). 그리퍼만 움직일 때는 팔 IK를 다시 풀지 않는다.
* Virtual 모드: `q_meas` 없이 `q_cmd`만 적분한다. 시작 자세는 HOME.

### 3.4 3D 뷰어 — 디버그 전용, 창 통합은 후순위

3D는 **XYZ/RPY가 맞는지 보는 모니터**다. 조그 버튼·숫자 필드가 입력이다. 한 창 임베드, 기즈모 드래그, ghost 로봇은 Physical Labs의 티칭 UX이고 이번 범위가 아니다.

**이번 버전 (1차에 구현)**

* 기동과 함께 Pinocchio `MeshcatVisualizer`를 띄운다 (별도 브라우저 창). Virtual/Real 모두 `q_cmd`가 3D에 보여야 한다. 3D 없이 GUI만 있는 상태를 완료로 치지 않는다. `package://` 메시 + TCP 축 프레임.
* 계약: `visualizer.display(q)` 만. controller는 Meshcat/Three/Qt를 모른다.
* 푸시: 최신 `q`만 **≤ 30 Hz**. 제어 루프가 더 빨라도 `display`를 틱마다 호출하지 않는다. 뷰어가 느려도 조그 루프를 막지 않는다 (드롭).
* 디버그 성공 기준: +X 조그 시 메시가 베이스 X로 움직이고, GUI의 mm/RPY와 FK가 같은 방향을 가리킨다.

**후순위 (하지 않음, 인터페이스만 막지 말 것)**

* GUI와 3D 한 창 (Qt WebEngine / 웹뷰). 툴킷 결정 후에.
* 3D에서 목표 포즈를 집어 `cmd`로 넣기.
* `tkinter` + `pywebview` 하이브리드로 한 창을 흉내 내기.

**하지 말 것**

* 뷰어 안에서 IK·충돌·시리얼.
* 3D 드래그를 실시간 조그처럼 쓰기.
* 표시용 pybullet GUI, matplotlib 3D.
* CustomTkinter 안에 Chromium을 억지로 넣기.

## 4. 좌표계 · 단위 · 캘리브레이션 (구현 전 고정)

| 항목 | 값 |
|---|---|
| 기준 프레임 | 로봇 베이스 (URDF `base_link` / world) |
| 위치 단위 | **mm** (내부 연산은 m, GUI·로그는 mm) |
| 자세 | **XYZ extrinsic RPY** (deg). `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)` |
| TCP | 기본값: 그리퍼 턱 끝 중점. `gripper_frame_link` 기준 고정 오프셋 (CAD, 예: 약 `[-8.3, 0, 5.8]` mm). `--tcp frame`이면 오프셋 0 |
| 조인트 단위 | 팔: URDF **도(deg)**. 그리퍼: **0–100** |
| HOME (URDF) | `shoulder_pan,lift,elbow_flex,elbow_roll,wrist_flex = 0`, `wrist_roll = -90`, `gripper = 50` |
| 워크스페이스 AABB (TCP, m) | 초기값 `x∈[-0.05,0.50]`, `y∈[-0.35,0.35]`, `z∈[-0.02,0.40]` (URDF에 맞게 조정 가능) |

**캘리브레이션 매핑 (실물 필수):** LeRobot 버스 각도 0°는 `range_min/max` 중점이 아니다. 캘리브 Enter 시점의 엔코더 중간값(STS3215 `2047`)이 URDF 0°이다. Connect 시 기존 `follower.json`의 `range_min`/`range_max`로 오프셋을 계산하고, 송수신 때마다 `bus ↔ URDF`를 변환한다. 재캘리브는 하지 않는다.

**모터 맵 (dof_mode=7):** ID 1..7 = `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `elbow_roll`, `wrist_flex`, `wrist_roll`, `gripper`. (`elbow_roll`이 ID 4, 이후 손목/그리퍼는 SO-101 대비 +1.)

## 5. 주요 기능 요구사항

### Task 1: URDF · Pinocchio 래퍼

* `SO101_6DOF.urdf`를 Pinocchio로 로드 (FK/IK + Meshcat). `package://` 메시 경로를 해석한다. 메시가 없으면 축/프레임만이라도 보이게 하고, 빈 화면을 성공으로 치지 않는다.
* 모델·data 초기화. 대상 프레임 `gripper_frame_link` + TCP 오프셋으로 위치/자코비안을 보정 (`p_tcp = p + R @ offset`, `Jv_tcp = Jv - [offset_world]× Jw`).
* FK, `local_world_aligned` 자코비안(병진·회전) API를 제공한다.
* Virtual 초기 `q` = HOME.

### Task 2: CustomTkinter GUI

한 화면에 관절(좌) / 카르테시안(우)을 둔다. 탭으로 숨기지 않는다.

* **카르테시안 조그:** +X −X +Y −Y +Z −Z +R −R +P −P +Yaw −Yaw. **누르는 동안만** 동작 (Press/Release). 버튼 밖에서 마우스를 떼도 정지.
* **그리퍼 조그:** Open / Close (누르는 동안).
* **관절 조그:** 6축 + 그리퍼 각각의 +/−.
* **절대 좌표:** TCP `X Y Z`(mm), `R P Y`(deg), gripper. [현재값 채우기], [이동]. 3D는 이 숫자의 결과를 보여줄 뿐 입력이 아니다.
* **절대 관절:** 각 관절 입력, [현재값 채우기], [이동].
* **상태:** Virtual Only / Real Hardware 스위치(또는 명시적 Connect). 현재 TCP(mm, RPY), **목표 vs 측정 오차(mm)**, 관절각, 모드, 폴트 문자열.
* **전역:** HOME, Stop jog, E-stop(토크 오프 + 명령 0).

조그 속도 초기값: 병진 40 mm/s, 회전 20 °/s, 관절 20 °/s, 그리퍼 40 unit/s. GUI에서 변경 가능하면 더 좋다.

### Task 3: 실시간 조그 및 IK

**하지 말 것:** 비감쇠 의사역행렬 \(J^{\dagger}\)만으로 조그. 원샷 LM/QP로 절대좌표에 점프.

**할 것:**

* **카르테시안 조그:** 버튼이 켜진 동안 목표 TCP를 속도만큼 적분 → 매 주기 DLS(감쇠 최소제곱)로 `q_cmd`를 목표에 서보. 특이점 근처에서 발산하지 않도록 `λ > 0`.
* **홀드 축:** 누르지 않은 XYZ는 누른 시점의 값을 1차 과제로 유지하고, 누른 축은 그 null space에서만 움직인다. 홀드 축 오차 0.5 mm 이하는 무시(엔코더 노이즈 추종 방지).
* **자세:** 병진 조그 중에는 누른 시점의 gripper tilt(roll/pitch)를 유지. yaw는 6축에서도 우선순위를 낮게. 회전 조그 중에는 해당 각속도를 적용. Z 단독 리프트 시 tilt 가중치를 줄여 어깨/엘보 관절이 굶지 않게 한다.
* **관절 조그:** IK 없이 `q`에 속도 적분. 관절 리밋에서 클램프.
* **명령 스무딩:** `q_send = (1-β) q_prev + β q_ik`. 카르테시안 스텝은 벡터 크기 제한 (축별 독립 클립 금지 — IK가 한 축을 굶김).
* **절대 좌표 이동:** 목표 SE3를 한 방에 `q*`로 보내지 않는다. 2–3초에 걸쳐 직선(위치) + 자세 보간하며 위와 같은 DLS 서보. 경로 각 스텝에 안전 가드.
* **절대 관절 이동:** 관절 공간 보간. IK 없음.
* LeRobot `max_relative_target`는 Real 연결 시 **끈다**. 안전은 카르테시안/관절 속도 상한으로 한다. (관절 상대클립은 카르테시안 IK를 깨뜨린다.)

DLS 튜닝 시작점(예제에서 검증됨, 6축에 맞게 재조정 가능): `λ ≈ 1.0`(mm 환산), 프레임당 관절 변화 상한 ~5°, 반복 12회 이내.

### Task 4: 안전 가드

제어 루프에서 `q_next`를 보내기 **전에** 적용. 위반 시 해당 속도/스텝을 버리고 `q` 유지, 상태줄에 이유 표시.

1. **TCP 바닥:** `z < z_floor` (기본 0, AABB `z_min`과 동일)로 내려가는 스텝 거부.
2. **워크스페이스 AABB:** TCP가 박스 밖으로 나가지 않게 목표를 클램프.
3. **관절 리밋:** 팔 ±180° (URDF/캘리브 범위가 더 좁으면 그 값). 그리퍼 0–100.
4. **속도 상한:** 카르테시안·관절 조그 속도 및 go-to 보간 속도.
5. **목표 리드:** 실물 추적보다 목표가 너무 앞서지 않게 (예: 20 mm, 15°).
6. **E-stop:** 즉시 조그/go-to 취소, 실물이면 토크 오프.
7. **자기충돌 / 엘보-테이블:** 1차 범위 밖. 명세에 존재만 명시하고 구현은 FK 기반 AABB 이후로 미뤄도 된다. TCP Z 가드만으로 엘보 충돌을 막았다고 쓰지 말 것.

### Task 5: 3D 디버그 뷰 (Meshcat) — **1차 범위**

첫 구현에 포함한다. 창 통합만 후순위다.

* 기동 시 Meshcat을 연다. **모드와 무관하게** `q_cmd`를 `display`한다. 푸시는 최신값 ≤ 30 Hz (§3.4).
* TCP 프레임(축)을 그려 GUI의 XYZ/RPY와 대조할 수 있게 한다.
* 창 배치(듀얼 모니터, 타일)는 사용자 몫. 앱이 GUI와 3D를 한 창에 넣지 않는다.
* 기즈모·ghost·임베드는 후순위. `display(q)` 계약만 유지하면 나중에 구현체를 바꿔도 된다.

### Task 6: 실물 하드웨어 · 모드 전환

`hw_controller.py`는 `SOFollower`(dof_mode=7, `use_degrees=True`)를 감싼다. Connect 시 `calibrate=False`, 기존 캘리브 로드, URDF 오프셋 계산, `max_relative_target=None`.

* **Virtual Only:** 시리얼 없음. 3D 뷰만. 초기 q = HOME.
* **Real Hardware 진입 순서 (필수):**
  1. 포트 연결, 엔코더 읽기 → `q_cmd = q_meas`로 **동기화** (Virtual에서 굴려 둔 q를 실물에 덤프하지 않음).
  2. 토크 온.
  3. 이후부터 `q_cmd` 전송.
* **Real → Virtual:** 토크 오프 여부는 GUI 옵션. `q_cmd`는 마지막 측정값 유지.
* **통신 실패 / Disconnect:** 루프 중단, 가능하면 토크 오프, UI에 에러. 재연결 전 다시 동기화.
* 가상과 실물의 “완벽 동기화”는 `q_cmd`를 둘 다에 넣는다는 뜻이다. 엔코더 지연·기어 백래시는 오차로 표시할 뿐, 강제로 맞추려 명령을 발산시키지 않는다.

포트 기본값 예: `/dev/so101_follower`, robot-id `follower`. CLI로 덮어쓴다.

## 6. 명시적 비범위 (이번 버전에 하지 말 것)

* ROS 2, MoveIt, RViz
* Feetech 패킷 파서 / 새 모터 SDK
* placo QP 원샷 IK를 카르테시안 조그의 주 솔버로 사용
* 카메라, 데이터셋 레코딩, 정책 추론
* 완전 자기충돌 / 메시 충돌 (후속)
* 뷰어 내부 IK, pybullet을 표시용으로 기동
* 제어 루프와 1:1로 동기된 3D 프레임 (뷰어는 드롭 가능)
* GUI·3D 한 창 통합, 3D 기즈모 입력, tk+웹뷰 하이브리드 (후순위)

## 7. 예상 산출물 (파일 구조)

```text
SO101_6DOF_pendant/
├── main.py                 # 진입점. 스레드 기동, CLI (port, urdf, tcp, dof-mode)
├── gui_app.py              # CustomTkinter. 명령만 생산, 상태만 표시
├── controller.py           # 30 Hz 루프. 조그/go-to/안전/모드. q 소유
├── robot_kinematics.py     # Pinocchio FK/J/DLS, TCP, 리밋
├── hw_controller.py        # SOFollower 래퍼, bus↔URDF 오프셋, 토크
├── visualizer.py           # display(q) — Meshcat 구현
├── SO101_6DOF.urdf              # (제공) 6-DOF + 메시
├── meshes/                 # URDF가 참조하는 STL/DAE (있으면)
└── requirements.txt        # pin, customtkinter, meshcat, numpy
```

LeRobot 저장소 안에 둘 경우 경로는 `examples/SO101_6DOF_pendant/` 도 허용한다. 그 경우 `hw_controller`는 `lerobot.robots.so_follower`를 import한다.

## 8. 완료 기준 (최소)

1. 로봇 없이 실행 → Meshcat에 HOME이 보이고, +X/−Y 조그 방향이 GUI mm·RPY와 일치한다. 창이 둘이어도 된다.
2. 실물 연결 시 팔이 Virtual의 마지막 자세로 튀지 않고, 현재 엔코더 자세에서 조그가 시작된다.
3. +X 조그 중 Y/Z가 수 mm 이상 흘러가지 않는다 (홀드 축).
4. TCP가 바닥 아래로 내려가는 조그는 무시되고 상태줄에 표시된다.
5. 그리퍼 Open/Close가 팔 자세를 건드리지 않는다.
6. HOME / Stop / E-stop이 동작한다.
7. GUI에 TCP mm, RPY, 관절각, 목표-측정 오차가 갱신된다.
8. 뷰어 갱신이 조그 루프를 막지 않는다 (`display`는 최신 `q` ≤ 30 Hz).
