"""펜던트 「픽앤플레이스(유저생성)」 버튼이 run(arm)을 호출한다.

arm은 펜던트가 이미 쓰는 팔이다. 이 파일에서 Arm()을 만들지 않는다.
동작이 끝난 뒤 time.sleep(1)은 다음 단계 전 1초 대기이다.
"""

import time

from motion import load_pick_targets


def run(arm) -> None:
    place_x, place_y = 150.0, 100.0
    # 카메라 창에 등록을 요청하고, 그 결과로 대기·집기 자세를 만든다.
    pick = load_pick_targets(arm)
    z_g = pick.g_xyz[2]

    arm.initial()
    time.sleep(1)
    arm.moveL(pick.p_xyz, "abs", rpy=pick.p_rpy)
    time.sleep(1)
    arm.grip(30)
    time.sleep(1)
    arm.moveL(pick.g_xyz, "abs", rpy=pick.g_rpy)
    time.sleep(1)
    arm.grip(0)
    time.sleep(1)
    arm.moveL([0, 0, 30], "rel")
    time.sleep(1)
    arm.moveL([place_x, place_y, z_g + 50], "abs", rpy=pick.place_rpy, path="joint")
    time.sleep(1)
    arm.moveL([place_x, place_y, z_g], "abs", rpy=pick.place_rpy)
    time.sleep(1)
    arm.grip(30)
    time.sleep(1)
    arm.moveL([0, 0, 50], "rel")
    time.sleep(1)
    arm.initial()
