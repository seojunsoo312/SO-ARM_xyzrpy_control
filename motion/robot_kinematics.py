"""Pinocchio wrapper: URDF load, FK, Jacobian, DLS servo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin

# URDF S1..S7 = Feetech ID 1..7 (jog-confirmed on Meshcat).
URDF_JOINT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
ARM_JOINT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6")
GRIPPER_JOINT = "S7"
# Fusion S7 q=0 = inner faces parallel (~16 mm pad gap).
# Tips meet at +11.5°. Opening is negative CAD.
# Pendant 0 = closed, 100 = fully open.
GRIPPER_CLOSED_CAD_DEG = 11.5
GRIPPER_OPEN_CAD_DEG = -120
GRIPPER_USER_MID = 50.0  # user deg 0 = bus/pendant 50 (HOME, half-open)
EE_FRAME = "L6_1"  # gripper body (wrist_roll child); not the moving jaw
TCP_FRAME = "tcp"
TCP_PARENT_JOINT = "S6"  # body-fixed; opening S7 must not move TCP
# Inner-pad midpoint of L6 (fixed jaw) and L7 (moving jaw) in the S6 frame, at HOME.
# Parent is S6 so opening S7 does not move TCP.
TCP_OFFSET_IN_L6 = np.array([0.00002, -0.10251, 0.00102])

# LeRobot names for later hw mapping. Jog-confirmed, chain order = servo ID.
LEROBOT_FROM_URDF = {
    "S1": "shoulder_pan",
    "S2": "shoulder_lift",
    "S3": "elbow_flex",
    "S4": "elbow_roll",
    "S5": "wrist_flex",
    "S6": "wrist_roll",
    "S7": "gripper",
}
URDF_FROM_LEROBOT = {v: k for k, v in LEROBOT_FROM_URDF.items()}

# CAD angle at pendant 0°: Onshape/LeRobot L-pose (upper arm up, forearm +X).
# Fusion URDF q=0 is the stretched CAD assembly along -Y, so these offsets
# make GUI 0 match that teaching zero (SO-101 screenshot, all joints 0.00).
JOINT_ZERO_OFFSET_DEG = {
    "S1": 0.0,
    "S2": -90.0,
    "S3": 90.0,
    "S4": 0.0,
    "S5": 0.0,
    "S6": 90.0,
    "S7": 0.0,
}

# Pendant + vs URDF + (CAD axis). -1 flips jog/display without moving HOME.
JOINT_SIGN = {
    "S1": -1.0,
    "S2": 1.0,
    "S3": 1.0,
    "S4": -1.0,  # elbow_roll: pendant + vs CAD was inverted
    "S5": 1.0,
    "S6": -1.0,
    "S7": -1.0,  # GUI + opens (CAD toward GRIPPER_OPEN_CAD_DEG)
}

# User-space HOME (deg). 0 = JOINT_ZERO_OFFSET_DEG in the URDF.
# Calib Enter at the L-pose makes user 0 that pose; do not add an extra S5 shift.
# Gripper HOME user 0 = pendant 50 (half-open); CAD is midway to GRIPPER_OPEN_CAD_DEG.
HOME_JOINTS_DEG = {name: 0.0 for name in URDF_JOINT_NAMES}

# Pendant "초기자세" — 작업 시작용. S7 user 0 = GUI 0–100 의 50.
INIT_POSE_JOINTS_DEG = {
    "S1": 0.0,
    "S2": -40.0,
    "S3": 40.0,
    "S4": 0.0,
    "S5": -90.0,
    "S6": 0.0,
    "S7": 0.0,
}

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def gripper_user_to_cad(user_deg: float) -> float:
    """User 0 = pendant 50. Maps pendant 0–100 onto CAD closed–open."""
    grip_100 = float(np.clip(user_deg + GRIPPER_USER_MID, 0.0, 100.0))
    span = GRIPPER_OPEN_CAD_DEG - GRIPPER_CLOSED_CAD_DEG
    return GRIPPER_CLOSED_CAD_DEG + (grip_100 / 100.0) * span


def gripper_cad_to_user(cad_deg: float) -> float:
    span = GRIPPER_OPEN_CAD_DEG - GRIPPER_CLOSED_CAD_DEG
    grip_100 = (float(cad_deg) - GRIPPER_CLOSED_CAD_DEG) / span * 100.0
    return float(np.clip(grip_100, 0.0, 100.0)) - GRIPPER_USER_MID

# Adjacent Fusion STLs overlap at the shared motor. Drop triangles inside this
# ball around the hinge, then collide the remaining link bodies.
ADJACENT_JOINT_CLEARANCE_M = 0.04

IK_ITERS = 12
MAX_CART_STEP_M = 0.006
MAX_HOLD_CORR_M = 0.0015
HOLD_DEADBAND_M = 0.002  # 2 mm. Tight hold on unused axes made S1/S4 hunt.
TILT_DEADBAND_RAD = 0.004
MAX_DQ_ITER_RAD = 1.5 * DEG2RAD
MAX_DQ_FRAME_RAD = 5.0 * DEG2RAD
DLS_LAMBDA = 2.0  # mm-equivalent. Higher → less nullspace chatter (S1/S4 on X).
DLS_LAMBDA_PRI = 0.4
TILT_MM_PER_RAD = 80.0
TILT_WEIGHT = 1.0
TILT_WEIGHT_EPS = 0.05
XYZ_TOL_M = 0.0003


@dataclass(frozen=True)
class TcpPose:
    xyz_m: np.ndarray  # (3,)
    rpy_deg: np.ndarray  # XYZ extrinsic, degrees
    rotation: np.ndarray  # 3x3

    @property
    def xyz_mm(self) -> np.ndarray:
        return self.xyz_m * 1000.0


def rpy_deg_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """XYZ extrinsic RPY (degrees): R = Rz @ Ry @ Rx."""
    rpy = np.deg2rad([float(roll), float(pitch), float(yaw)])
    return np.asarray(pin.rpy.rpyToMatrix(rpy), dtype=float)


def rotmat_to_rpy_deg(rot: np.ndarray) -> np.ndarray:
    """XYZ extrinsic RPY (degrees), inverse of rpy_deg_to_rotmat."""
    rpy = pin.rpy.matrixToRpy(np.asarray(rot, dtype=float))
    return np.degrees(np.asarray(rpy, dtype=float)).reshape(3)


def tilt_weight_from_delta(delta_xyz: np.ndarray) -> float:
    """Full orientation hold for XY; fade on pure Z (same joints as pitch)."""
    d = np.asarray(delta_xyz, dtype=float).reshape(3)
    xy = float(np.hypot(d[0], d[1]))
    z = abs(float(d[2]))
    return float(TILT_WEIGHT * xy / (xy + z + 1e-9))


def _dls_dq(jac: np.ndarray, err: np.ndarray, *, damping: float, max_dq: float) -> np.ndarray:
    j = np.asarray(jac, dtype=float)
    e = np.asarray(err, dtype=float)
    if j.size == 0 or e.size == 0:
        return np.zeros(j.shape[1] if j.ndim == 2 else 0)
    n = j.shape[1]
    lam = float(max(damping, 0.0))
    dq = np.linalg.solve(j.T @ j + (lam * lam) * np.eye(n), j.T @ e)
    peak = float(np.max(np.abs(dq))) if dq.size else 0.0
    if peak > max_dq > 0:
        dq = dq * (max_dq / peak)
    return dq


def _hierarchical_dq(
    j_pri: np.ndarray,
    e_pri: np.ndarray,
    j_sec: np.ndarray,
    e_sec: np.ndarray,
    *,
    n_joints: int,
    max_dq: float,
) -> np.ndarray:
    if j_pri.size == 0:
        return _dls_dq(j_sec, e_sec, damping=DLS_LAMBDA, max_dq=max_dq)
    dq1 = _dls_dq(j_pri, e_pri, damping=DLS_LAMBDA_PRI, max_dq=0.0)
    if j_sec.size == 0:
        dq = dq1
    else:
        lam2 = DLS_LAMBDA_PRI * DLS_LAMBDA_PRI
        jpinv = np.linalg.solve(j_pri.T @ j_pri + lam2 * np.eye(n_joints), j_pri.T)
        e2 = e_sec - j_sec @ dq1
        dq2 = _dls_dq(j_sec @ (np.eye(n_joints) - jpinv @ j_pri), e2, damping=DLS_LAMBDA, max_dq=0.0)
        dq = dq1 + dq2
    peak = float(np.max(np.abs(dq))) if dq.size else 0.0
    if peak > max_dq > 0:
        dq = dq * (max_dq / peak)
    return dq


def _clip_position_error(err_p: np.ndarray, cmd_mask: np.ndarray | None) -> np.ndarray:
    err_p = np.asarray(err_p, dtype=float).copy()
    if cmd_mask is None:
        nerr = float(np.linalg.norm(err_p))
        if nerr > MAX_CART_STEP_M:
            err_p = err_p * (MAX_CART_STEP_M / nerr)
        return err_p
    cmd = np.asarray(cmd_mask, dtype=bool).reshape(3)
    hold = ~cmd
    if np.any(cmd):
        e_cmd = np.where(cmd, err_p, 0.0)
        n_cmd = float(np.linalg.norm(e_cmd))
        if n_cmd > MAX_CART_STEP_M:
            err_p[cmd] = e_cmd[cmd] * (MAX_CART_STEP_M / n_cmd)
    if np.any(hold):
        err_p[hold] = np.where(np.abs(err_p[hold]) < HOLD_DEADBAND_M, 0.0, err_p[hold])
        e_hold = np.where(hold, err_p, 0.0)
        n_hold = float(np.linalg.norm(e_hold))
        if n_hold > MAX_HOLD_CORR_M:
            err_p[hold] = e_hold[hold] * (MAX_HOLD_CORR_M / n_hold)
    return err_p


class RobotKinematics:
    """Load SO101_6DOF.urdf; FK + DLS Cartesian servo. Hardware mapping is elsewhere."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        package_dirs: list[str] | None = None,
        tcp_offset: np.ndarray | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        if package_dirs is None:
            # package://meshes/*.stl → look next to the URDF, then its parent.
            package_dirs = [str(self.urdf_path.parent), str(self.urdf_path.parent.parent)]

        self.tcp_offset = (
            np.asarray(TCP_OFFSET_IN_L6, dtype=float).reshape(3)
            if tcp_offset is None
            else np.asarray(tcp_offset, dtype=float).reshape(3)
        )

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        missing = [n for n in URDF_JOINT_NAMES if not self.model.existJointName(n)]
        if missing:
            raise ValueError(f"URDF missing joints {missing}; have {self.model.names}")
        if not self.model.existFrame(EE_FRAME):
            raise ValueError(f"URDF missing EE frame {EE_FRAME!r}")
        self._add_tcp_frame()
        self.data = self.model.createData()
        self.visual_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.VISUAL,
            package_dirs=package_dirs,
        )
        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=package_dirs,
        )
        self._init_collision_pairs()
        self.ee_frame_id = self.model.getFrameId(TCP_FRAME)

        # q / v index for each URDF joint (skip universe at names[0]).
        self._q_index: dict[str, int] = {}
        self._v_index: dict[str, int] = {}
        for name in URDF_JOINT_NAMES:
            jid = self.model.getJointId(name)
            self._q_index[name] = int(self.model.idx_qs[jid])
            self._v_index[name] = int(self.model.idx_vs[jid])
        self._arm_v_cols = [self._v_index[n] for n in ARM_JOINT_NAMES]
        self._arm_q_cols = [self._q_index[n] for n in ARM_JOINT_NAMES]

    def _add_tcp_frame(self) -> None:
        jid = self.model.getJointId(TCP_PARENT_JOINT)
        placement = pin.SE3(np.eye(3), self.tcp_offset)
        self.model.addFrame(pin.Frame(TCP_FRAME, jid, placement, pin.FrameType.OP_FRAME))

    def _init_collision_pairs(self) -> None:
        """Skip-1+ full meshes, plus adjacent fold using hinge-clearance bodies."""
        cm = self.collision_model
        cm.addAllCollisionPairs()
        keep: list[tuple[int, int]] = []
        for pair in cm.collisionPairs:
            ja = int(cm.geometryObjects[pair.first].parentJoint)
            jb = int(cm.geometryObjects[pair.second].parentJoint)
            if abs(ja - jb) <= 1:
                continue
            keep.append((int(pair.first), int(pair.second)))
        cm.removeAllCollisionPairs()
        for i, j in keep:
            cm.addCollisionPair(pin.CollisionPair(i, j))
        self.collision_data = pin.GeometryData(cm)
        self._adj_body_pairs = self._build_adjacent_body_pairs()

    def _build_adjacent_body_pairs(self) -> list[tuple[int, int, object, object]]:
        """Link-body BVHs with the shared motor CAD cut out. BASE–L1 skipped (pan)."""
        cm = self.collision_model
        pin.forwardKinematics(self.model, self.data, pin.neutral(self.model))
        gd = pin.GeometryData(cm)
        pin.updateGeometryPlacements(self.model, self.data, cm, gd)
        n = len(cm.geometryObjects)
        cache: dict[tuple[int, int], object] = {}
        pairs: list[tuple[int, int, object, object]] = []
        for i in range(n):
            for j in range(i + 1, n):
                ja = int(cm.geometryObjects[i].parentJoint)
                jb = int(cm.geometryObjects[j].parentJoint)
                # BASE–L1 pan is CAD-overlap; L6–L7 jaws are meant to close.
                if abs(ja - jb) != 1 or min(ja, jb) == 0:
                    continue
                if max(ja, jb) == int(self.model.getJointId(GRIPPER_JOINT)):
                    continue
                jid = max(ja, jb)
                bodies = []
                for gi in (i, j):
                    key = (gi, jid)
                    if key not in cache:
                        cache[key] = self._body_bvh_far_from_joint(
                            cm.geometryObjects[gi], gd.oMg[gi], jid
                        )
                    bodies.append(cache[key])
                if bodies[0] is None or bodies[1] is None:
                    continue
                pairs.append((i, j, bodies[0], bodies[1]))
        return pairs

    def _body_bvh_far_from_joint(self, go: pin.GeometryObject, oMg: pin.SE3, jid: int):
        geom = go.geometry
        verts = np.asarray(geom.vertices(), dtype=float)
        joint_local = np.asarray(
            oMg.inverse().act(np.asarray(self.data.oMi[jid].translation, dtype=float))
        )
        far = np.linalg.norm(verts - joint_local, axis=1) >= ADJACENT_JOINT_CLEARANCE_M
        kept: list[tuple[int, int, int]] = []
        for k in range(int(geom.num_tris)):
            tri = geom.tri_indices(k)
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            if far[a] and far[b] and far[c]:
                kept.append((a, b, c))
        if len(kept) < 8:
            return None
        used = sorted({idx for tri in kept for idx in tri})
        remap = {old: new for new, old in enumerate(used)}
        pts = coal.StdVec_Vec3s()
        for idx in used:
            pts.append(np.asarray(verts[idx], dtype=float))
        tris = coal.StdVec_Triangle()
        for a, b, c in kept:
            tris.append(coal.Triangle(remap[a], remap[b], remap[c]))
        bvh = coal.BVHModelOBBRSS()
        bvh.beginModel(len(tris), len(pts))
        bvh.addSubModel(pts, tris)
        bvh.endModel()
        bvh.computeLocalAABB()
        return bvh

    def in_collision(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=float)
        if pin.computeCollisions(
            self.model, self.data, self.collision_model, self.collision_data, q, True
        ):
            return True
        return self._adjacent_bodies_in_collision(q)

    def _adjacent_bodies_in_collision(self, q: np.ndarray) -> bool:
        pin.forwardKinematics(self.model, self.data, q)
        gd = pin.GeometryData(self.collision_model)
        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, gd)
        req = coal.CollisionRequest()
        for i, j, bi, bj in self._adj_body_pairs:
            ti, tj = gd.oMg[i], gd.oMg[j]
            oi = coal.CollisionObject(
                bi, coal.Transform3s(np.asarray(ti.rotation), np.asarray(ti.translation))
            )
            oj = coal.CollisionObject(
                bj, coal.Transform3s(np.asarray(tj.rotation), np.asarray(tj.translation))
            )
            res = coal.CollisionResult()
            coal.collide(oi, oj, req, res)
            if res.isCollision():
                return True
        return False

    def q_index(self, joint_name: str) -> int:
        return self._q_index[joint_name]

    def joints_deg(self, q: np.ndarray) -> dict[str, float]:
        """Pendant degrees = URDF degrees minus the L-pose zero offset."""
        out: dict[str, float] = {}
        for name in URDF_JOINT_NAMES:
            cad = float(q[self._q_index[name]] * RAD2DEG)
            if name == GRIPPER_JOINT:
                out[name] = gripper_cad_to_user(cad)
            else:
                out[name] = JOINT_SIGN[name] * (cad - JOINT_ZERO_OFFSET_DEG[name])
        return out

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        """Clip gripper to closed–open CAD; other joints unbounded."""
        q = np.asarray(q, dtype=float).copy()
        i = self._q_index[GRIPPER_JOINT]
        lo = min(GRIPPER_CLOSED_CAD_DEG, GRIPPER_OPEN_CAD_DEG) * DEG2RAD
        hi = max(GRIPPER_CLOSED_CAD_DEG, GRIPPER_OPEN_CAD_DEG) * DEG2RAD
        q[i] = float(np.clip(q[i], lo, hi))
        return q

    def q_home(self) -> np.ndarray:
        return self.q_from_deg(HOME_JOINTS_DEG)

    def q_from_deg(self, joints_deg: dict[str, float]) -> np.ndarray:
        """Build Pinocchio q from pendant (user) degrees."""
        q = pin.neutral(self.model).copy()
        for name in URDF_JOINT_NAMES:
            user = float(joints_deg.get(name, HOME_JOINTS_DEG[name]))
            if name == GRIPPER_JOINT:
                cad = gripper_user_to_cad(user)
            else:
                cad = JOINT_ZERO_OFFSET_DEG[name] + JOINT_SIGN[name] * user
            q[self._q_index[name]] = cad * DEG2RAD
        return q

    def forward_tcp(self, q: np.ndarray) -> TcpPose:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        rot = np.asarray(oMf.rotation, dtype=float).copy()
        xyz = np.asarray(oMf.translation, dtype=float).copy()
        return TcpPose(xyz_m=xyz, rpy_deg=rotmat_to_rpy_deg(rot), rotation=rot)

    def tcp_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """TCP pose and world Jacobian of the 6 arm joints. Jv: m/rad, Jw: 1/rad."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        jac = np.asarray(
            pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            ),
            dtype=float,
        )[:, self._arm_v_cols]
        return (
            np.asarray(oMf.translation, dtype=float).copy(),
            np.asarray(oMf.rotation, dtype=float).copy(),
            jac[0:3].copy(),
            jac[3:6].copy(),
        )

    def servo_toward(
        self,
        q: np.ndarray,
        target_xyz: np.ndarray,
        *,
        R_ref: np.ndarray | None = None,
        cmd_mask: np.ndarray | None = None,
        ori_weight: float = TILT_WEIGHT,
    ) -> np.ndarray:
        """One control-frame of DLS IK. Returns a new full `q` (gripper unchanged)."""
        q_out = np.asarray(q, dtype=float).copy()
        q_arm = q_out[self._arm_q_cols].copy()
        q_start = q_arm.copy()
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        cmd = None if cmd_mask is None else np.asarray(cmd_mask, dtype=bool).reshape(3)
        n_arm = len(ARM_JOINT_NAMES)
        use_ori = R_ref is not None and ori_weight >= TILT_WEIGHT_EPS
        for _ in range(IK_ITERS):
            pos, rot, jv, jw = self.tcp_jacobian(self._q_with_arm(q_out, q_arm))
            err_p = _clip_position_error(target_xyz - pos, cmd)
            nerr = float(np.linalg.norm(target_xyz - pos))
            jac_p = jv * 1000.0
            err_p_mm = err_p * 1000.0
            j_extra: np.ndarray | None = None
            e_extra: np.ndarray | None = None
            if use_ori:
                assert R_ref is not None
                err_r = np.clip(np.asarray(pin.log3(R_ref @ rot.T)).reshape(3), -0.04, 0.04)
                err_r = np.where(np.abs(err_r) < TILT_DEADBAND_RAD, 0.0, err_r)
                if nerr < XYZ_TOL_M and float(np.linalg.norm(err_r)) < 0.002:
                    break
                tw = float(ori_weight)
                j_extra = jw * TILT_MM_PER_RAD * tw
                e_extra = err_r * TILT_MM_PER_RAD * tw
            elif nerr < XYZ_TOL_M:
                break
            if cmd is None:
                if j_extra is None:
                    dq = _dls_dq(jac_p, err_p_mm, damping=DLS_LAMBDA, max_dq=MAX_DQ_ITER_RAD)
                else:
                    dq = _dls_dq(
                        np.vstack([jac_p, j_extra]),
                        np.concatenate([err_p_mm, e_extra]),
                        damping=DLS_LAMBDA,
                        max_dq=MAX_DQ_ITER_RAD,
                    )
            else:
                # One DLS on XYZ. Hold-first hierarchy leaves Z in a tiny leftover
                # of Jxy (table-fold pose: S2/S3 busy holding XY) so +Z looks dead.
                if j_extra is None:
                    dq = _dls_dq(jac_p, err_p_mm, damping=DLS_LAMBDA, max_dq=MAX_DQ_ITER_RAD)
                else:
                    dq = _dls_dq(
                        np.vstack([jac_p, j_extra]),
                        np.concatenate([err_p_mm, e_extra]),
                        damping=DLS_LAMBDA,
                        max_dq=MAX_DQ_ITER_RAD,
                    )
            q_arm = q_arm + dq
            if float(np.max(np.abs(q_arm - q_start))) >= MAX_DQ_FRAME_RAD:
                break
        q_out[self._arm_q_cols] = q_arm
        return q_out

    def _q_with_arm(self, q_full: np.ndarray, q_arm: np.ndarray) -> np.ndarray:
        out = np.asarray(q_full, dtype=float).copy()
        out[self._arm_q_cols] = q_arm
        return out
