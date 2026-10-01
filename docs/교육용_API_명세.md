# 교육용 팔 API 명세서

학생이 역기구학·시리얼·펜던트 GUI를 다루지 않고, TCP를 mm와 deg로 움직이는 계약이다. 구현은 `motion.Arm`과 `load_pick_targets`이다.

## 1. 범위

- 대상: SO-101 기반 6자유도 팔 + 그리퍼 1축.
- 학생 코드는 `Arm`과 `load_pick_targets`만 호출한다. IK와 모터 버스는 이 API 밖에 둔다.
- 두 가지 모드가 있다.
  - `virtual` (기본): 로봇 없이 Meshcat만 움직인다.
  - `real`: 실기 시리얼에 연결한다. 연결에 실패하면 시작 시점에 예외가 난다.
- 시작과 함께 Meshcat이 브라우저에 뜬다. 주소는 콘솔에 출력된다.

이 API에 없는 것: 조그, 카메라 화면. 속도는 `moveL`·`moveJ` 인자로만 준다. `load_pick_targets`는 카메라 창에 물체 등록을 다시 요청한다.

## 2. 시작과 종료

```python
from motion import Arm, load_pick_targets

arm = Arm()                 # virtual
arm = Arm(mode="real")      # 실기
pick = load_pick_targets()  # 카메라에 등록을 요청한 뒤 대기·집기·놓기 자세
arm.close()
```

`mode`는 `"virtual"` 또는 `"real"`만 받는다. 그 외 문자열은 `ArmError`이다.

`close()`는 제어 루프를 멈추고, 실기이면 연결을 끊는다. 종료된 팔의 메서드는 `ArmError`이다.

## 3. 좌표와 단위

숫자의 기준은 로봇 베이스이다.

| 항목 | 값 |
|---|---|
| +X | 앞 |
| +Y | 왼쪽 |
| +Z | 위 |
| 위치 | mm |
| 자세 | roll, pitch, yaw (deg) |
| 자세 순서 | XYZ 외부 회전. `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)` |
| TCP | 그리퍼 턱 끝의 중점 |

`where()`가 돌려주는 위치가 이 좌표이다. 자세를 생략한 `moveL`은 지금 자세를 유지한다. 관절 0은 홈인 L자이다.

## 4. 명령

호출은 이동이 끝나거나 실패할 때까지 그 줄에서 기다린다. `stop`과 `close`만 즉시 반환한다.

| 명령 | 의미 | 속도 | 반환 |
|---|---|---|---|
| `arm.where()` | 지금 관절, 그리퍼, TCP | 없음 | 상태 딕셔너리 |
| `arm.moveL([dx, dy, dz], "rel")` | 지금 위치에서 mm만큼. 자세 유지 | `speed` 기본 25 mm/s (5–45) | 도착 후 상태 |
| `arm.moveL([x, y, z], "abs")` | 베이스 절대 mm. 자세 유지 | `speed` 기본 25 mm/s (5–45) | 도착 후 상태 |
| `arm.moveL([x, y, z], "abs", rpy=[r, p, y])` | 절대 mm와 자세. TCP 직선 | `speed` 기본 25 mm/s (5–45), `rpy_speed` 기본 25 deg/s (5–45) | 도착 후 상태 |
| `arm.moveL([x, y, z], "abs", rpy=[r, p, y], path="joint")` | 절대 mm와 자세. 관절 이동 | `speed` 기본 24 deg/s (8–45). `rpy_speed` 없음 | 도착 후 상태 |
| `arm.moveJ([j1..j6], "abs")` | 관절 절대각. 그리퍼 유지 | `speed` 기본 24 deg/s (8–45) | 도착 후 상태 |
| `arm.moveJ([dj1..dj6], "rel", gripper=20)` | 관절은 지금 각에 더함. 그리퍼도 더함 | `speed` 기본 24 deg/s (8–45) | 도착 후 상태 |
| `arm.grip(0~100)` | 0=닫힘, 100=열림 | 인자 없음. 관절 기본 24°/s | 없음 |
| `arm.home()` | 홈 자세 | 인자 없음. 관절 기본 24°/s | 도착 후 상태 |
| `arm.initial()` | 펜던트 초기자세 | 인자 없음. 관절 기본 24°/s | 도착 후 상태 |
| `arm.stop()` | 이동 중단 | 없음 | 없음 |
| `arm.close()` | 루프·실기 연결 종료 | 없음 | 없음 |

상태 딕셔너리:

```python
{
    "joints_deg": [j1, j2, j3, j4, j5, j6],
    "gripper": 50,
    "xyz_mm": [x, y, z],
    "rpy_deg": [roll, pitch, yaw],
}
```

`j1`–`j6`는 S1–S6, 단위는 도이다. `gripper`는 0이 닫힘, 100이 열림이다.

각 값은 소수 둘째 자리까지이다.

### 4.1 `moveL`

```python
arm.moveL(xyz, mode, rpy=None, speed=None, rpy_speed=None, path="line")
```

- `xyz`: 길이 3. 단위 mm. 유한한 숫자만 받는다.
- `mode`: `"abs"` 또는 `"rel"`.
  - `"rel"`: `xyz`는 지금 위치에 더하는 변위이다.
  - `"abs"`: `xyz`는 베이스 기준 목표 위치이다.
- `rpy`: 생략하면 지금 roll, pitch, yaw를 유지한다. 주면 목표 자세(deg)이다. `"rel"`과 `"abs"` 모두에서 쓸 수 있다. `rpy`는 변위가 아니라 목표 자세이다.
- `path`: `"line"` 또는 `"joint"`. 생략하면 `"line"`.
  - `"line"`: 그리퍼 끝이 직선으로 간다. 직선이 팔과 겹치면 위로 올렸다가 간다.
  - `"joint"`: 목표 좌표를 지금 팔 자세 근처의 관절각으로 바꾼 뒤, `moveJ`처럼 관절을 돌린다. 끝의 경로는 그때 도는 관절에 따라 달라진다.
- `speed`: `path="line"`이면 직선 속도(mm/s). 생략하면 25. 범위는 5–45이다. `path="joint"`이면 관절 속도(deg/s). 생략하면 24. 범위는 8–45이다. 가장 많이 움직이는 관절 기준으로 시간을 잡는다.
- `rpy_speed`: 자세 속도(deg/s). `path="line"`에서만 쓴다. 생략하면 25. 범위는 5–45이다. 자세를 유지하는 이동에는 쓰이지 않는다. `path="joint"`에서 주면 `ArmError`이다.

도달할 수 없거나 충돌로 막히면 `ArmError`이고, 그 호출은 목표에 도착한 것으로 치지 않는다.

### 4.2 `moveJ`

```python
arm.moveJ(joints, mode, gripper=None, speed=None)
```

- `joints`: 길이 6. S1–S6, 단위는 도이다.
- `mode`: `"abs"` 또는 `"rel"`.
  - `"rel"`: `joints`는 지금 관절각에 더하는 도이다.
  - `"abs"`: `joints`는 목표 관절각이다.
- `gripper`: 생략하면 그리퍼는 움직이지 않는다. 넣으면 같은 `mode`를 따른다. `"abs"`는 0–100으로 맞추고, `"rel"`은 지금 값에 더한다. 결과는 0과 100 안으로 자른다. 관절에 더하는 값은 도이고, 그리퍼에 더하는 값은 0–100 눈금이다.
- `speed`: 관절 속도(deg/s). 생략하면 24. 범위는 8–45이다. 가장 많이 움직이는 관절 기준으로 시간을 잡는다.

`moveJ`는 관절 사이 보간이다. TCP 직선 이동이 아니다.

### 4.3 `grip`

`0`은 닫힘, `100`은 완전히 열림이다. 범위 밖 값은 0과 100 안으로 자른다. 그리퍼만 움직이고 TCP 목표는 바꾸지 않는다. 속도 인자는 없으며 관절 보통값 24°/s 쪽 기본 속도를 쓴다. `home()`, `initial()`도 같다.

### 4.4 `home`과 `initial`

둘 다 관절 목표로 이동한다. Cartesian 직선 이동이 아니다.

- `home`: 관절각 0. 위팔이 서고 아래팔이 앞(+X)을 향하는 L자. 그리퍼는 반쯤 열림.
- `initial`: 펜던트 초기자세. 작업 시작용이며 그리퍼는 닫힘.

### 4.5 `stop`

진행 중인 이동을 멈춘다. `moveL`, `moveJ`, `grip`, `home`, `initial`은 끝날 때까지 그 줄을 붙잡고 있으므로, 같은 흐름에서 그 호출 다음에 적은 `stop`은 이동이 끝난 뒤에 실행된다. 이동 중에 끊으려면 다른 실행 흐름에서 `stop`을 호출한다.

### 4.6 `load_pick_targets`

```python
pick = load_pick_targets()
pick = load_pick_targets(arm)
```

펜던트 **물체 위치 불러오기**와 같이, 카메라 창(`roi_cloud`)에 물체 등록을 요청한다. 그 6D가 올 때까지 이 줄에서 기다린 뒤 픽앤플레이스 좌표를 만든다. 저장된 자세 파일을 읽지 않는다. 카메라 창이 켜져 있어야 한다. 초록 버튼과 동시에 호출하면 등록 요청이 겹친다.

반환 필드는 아래와 같다. 각 값은 소수 둘째 자리이다.

| 필드 | 의미 |
|---|---|
| `p_xyz`, `p_rpy` | 대기 위치 P (mm, deg) |
| `g_xyz`, `g_rpy` | 집기 위치 G (mm, deg) |
| `place_rpy` | 놓기 자세 (deg). `g_rpy`로 정해진다 |

집기 높이 `z_g`는 `pick.g_xyz[2]`이다. 내려놓을 X, Y는 이 함수에 없다. 실습 코드에서 상수로 둔다. 기본 예는 `(150, -100)`이다.

`arm`을 넘기면 그 팔의 현재 관절로 손목 자세를 고른다. 넘기지 않으면 홈 관절을 쓴다. 카메라 창이 응답하지 않거나, 물체 박스가 없거나, 등록에 실패하면 `ArmError`이다.

## 5. 오류

실패는 `ArmError`이다. 학생 코드는 이 예외만 보면 된다.

대표적인 경우:

- `mode`가 `"virtual"`, `"real"`이 아님
- 실기 연결 실패
- `moveL` 또는 `moveJ`의 `mode`가 `"abs"`, `"rel"`이 아님
- `xyz` 또는 `rpy`가 숫자 3개가 아님
- `joints`가 숫자 6개가 아님
- `gripper`가 숫자가 아님
- `moveL`의 `path`가 `"line"`, `"joint"`가 아님
- `moveL`의 `path="line"`인데 `speed`가 5–45 mm/s가 아님, `rpy_speed`가 5–45 deg/s가 아님
- `moveL`의 `path="joint"`인데 `speed`가 8–45 deg/s가 아님, 또는 `rpy_speed`를 줌
- `moveJ`의 `speed`가 8–45 deg/s가 아님
- 이미 `close()`된 팔
- 목표에 도달하지 못함 (IK 실패, 충돌, 이동 시간 초과)
- `load_pick_targets()`인데 카메라 창이 응답하지 않음, 바운딩 박스가 없음, 등록 실패, 또는 이미 등록 중

## 6. 예

예시는 가상 팔 기준이다. 실기는 마지막 예만 모드가 다르다.

```python
from motion import Arm

# 예: 가상 팔 시작. 브라우저에 Meshcat이 뜬다.
arm = Arm()

# 예: 지금 관절, 그리퍼, TCP.
print(arm.where())
```

```python
# 예: 작업 시작 자세. 그리퍼는 닫혀 있다.
arm.initial()

# 예: 그리퍼를 연다. 100이 완전히 열림.
arm.grip(100)

# 예: 상대 이동. 지금 위치에서 앞(+X)으로 30mm. 자세는 유지. 직선 10 mm/s.
arm.moveL([30, 0, 0], "rel", speed=10)

# 예: 상대 이동. 왼쪽(+Y) 20mm, 위(+Z) 15mm.
arm.moveL([0, 20, 15], "rel")

# 예: 절대 이동. 베이스 기준 x=200, y=0, z=120 mm. 자세는 유지.
arm.moveL([200, 0, 120], "abs")

# 예: 위치는 그대로 두고 자세만 목표로. rpy는 변위가 아니다.
arm.moveL([0, 0, 0], "rel", rpy=[0, 90, 0])

# 예: 절대 위치와 자세를 함께 지정.
arm.moveL([180, 40, 90], "abs", rpy=[0, 80, 0])

# 예: 같은 목표를 관절 이동으로. 그리퍼 끝은 직선이 아닐 수 있다.
arm.moveL([180, 40, 90], "abs", rpy=[0, 80, 0], path="joint")
```

```python
# 예: where()로 읽은 높이에 40mm를 더해 절대 좌표로 올린다.
here = arm.where()
x, y, z = here["xyz_mm"]
arm.moveL([x, y, z + 40], "abs")

# 예: 반쯤 열기. 0이 닫힘, 50이 중간, 100이 완전히 열림.
arm.grip(50)

# 예: 관절 절대 이동. S1–S6 도. 그리퍼는 그대로.
arm.moveJ([0, -40, 40, 0, -90, 0], "abs")

# 예: 2번 관절만 +10°. 그리퍼는 지금보다 +20.
arm.moveJ([0, 10, 0, 0, 0, 0], "rel", gripper=20)
```

```python
# 예: 카메라에 등록을 요청한 뒤 집어 드롭 위치에 놓는다.
# 대기·집기·놓기 자세는 load_pick_targets()가 채운다. 카메라 창이 켜져 있어야 한다.
drop_x, drop_y = 150.0, -100.0
pick = load_pick_targets()
z_g = pick.g_xyz[2]

arm.initial()
arm.moveL(pick.p_xyz, "abs", rpy=pick.p_rpy)
arm.grip(30)
arm.moveL(pick.g_xyz, "abs", rpy=pick.g_rpy)
arm.grip(0)
arm.moveL([0, 0, 30], "rel")
arm.moveL([drop_x, drop_y, z_g + 50], "abs", rpy=pick.place_rpy)
arm.moveL([drop_x, drop_y, z_g], "abs", rpy=pick.place_rpy)
arm.grip(30)
arm.moveL([0, 0, 50], "rel")
arm.initial()
arm.close()
```

```python
# 예: 집어서 옆으로 내려놓는 순서. 좌표는 직접 적는다.
arm.initial()
arm.grip(100)                      # 접근 전에 연다
arm.moveL([0, 0, -40], "rel")       # 물체 쪽으로 내린다
arm.grip(0)                        # 집는다
arm.moveL([0, 0, 50], "rel")        # 들어 올린다
arm.moveL([0, 80, 0], "rel")        # 왼쪽으로 옮긴다
arm.moveL([0, 0, -40], "rel")       # 내려놓기 위해 내린다
arm.grip(100)                      # 놓는다
arm.moveL([0, 0, 40], "rel")        # 빠진다
arm.home()                         # L자 홈. 그리퍼는 반쯤 열림
arm.close()                        # 루프 종료
```

```python
# 예: 실기. 포트에 팔이 연결되어 있어야 한다.
arm = Arm(mode="real")
print(arm.where())
arm.close()
```

## 7. 에러 케이스

실패하면 `ArmError`이다. 인자를 빼먹으면 `TypeError`이다. 아래는 그 호출과 메시지이다.

```python
# (에러케이스) mode 오타. "simulation", "hw" 도 같은 예외.
Arm(mode="sim")
# ArmError: mode는 'virtual' 또는 'real' 이어야 합니다
```

```python
# (에러케이스) 실기인데 팔이 없거나 포트가 안 열린다.
Arm(mode="real")
# ArmError: 실기 연결 실패 (...)
```

```python
# (에러케이스) mode 문자열을 빼먹음.
arm.moveL([30, 0, 0])
# TypeError: moveL() missing 1 required positional argument: 'mode'
```

```python
# (에러케이스) 좌표와 mode 순서가 바뀜.
arm.moveL("abs", [200, 0, 80])
# ArmError: moveL의 두 번째 인자는 'abs' 또는 'rel' 이어야 합니다
```

```python
# (에러케이스) "relative", "absolute" 는 받지 않는다.
arm.moveL([30, 0, 0], "relative")
# ArmError: moveL의 두 번째 인자는 'abs' 또는 'rel' 이어야 합니다
```

```python
# (에러케이스) 좌표를 리스트가 아니라 숫자 세 개로 전달. 두 번째 숫자가 mode가 된다.
arm.moveL(30, 0, 0)
# ArmError: moveL의 두 번째 인자는 'abs' 또는 'rel' 이어야 합니다
```

```python
# (에러케이스) xyz가 3개가 아님.
arm.moveL([30, 0], "rel")
# ArmError: xyz는 mm 3개입니다
```

```python
# (에러케이스) rpy가 3개가 아님.
arm.moveL([200, 0, 80], "abs", rpy=[0, 90])
# ArmError: rpy는 deg 3개입니다
```

```python
# (에러케이스) mm 자리에 m를 넣음. 0.2는 0.2mm라 베이스 한가운데로 가는 명령이 된다.
arm.moveL([0.2, 0.0, 0.08], "abs")
# ArmError: IK 실패 또는 충돌, 이동이 막힘
```

```python
# (에러케이스) 팔이 닿지 않는 절대 좌표.
arm.moveL([800, 0, 100], "abs")
# ArmError: IK 실패 (... mm) 또는 go-to leftover / go-to blocked
```

```python
# (에러케이스) 상대 이동을 절대 좌표처럼 사용. 지금 위치에 200, 0, 80을 더한다.
arm.moveL([200, 0, 80], "rel")
# 작업 공간 밖이면 ArmError: IK 실패 또는 go-to blocked
```

```python
# (에러케이스) rpy를 지금 자세에 더하는 변위로 사용.
# 지금 yaw가 90이면 목표는 100이 아니라 10이다. 자세 변화가 크면 도달에 실패한다.
arm.moveL([0, 0, 0], "rel", rpy=[0, 0, 10])
# ArmError: IK 실패 또는 go-to leftover
```

```python
# (에러케이스) 카메라 창이 꺼져 있어 등록 요청에 답이 없음.
load_pick_targets()
# ArmError: 카메라 창이 응답하지 않습니다. 실행중인지 확인해주세요
```

```python
# (에러케이스) 끝난 팔에 다시 명령.
arm.close()
arm.where()
# ArmError: 이미 종료된 Arm 입니다
```
