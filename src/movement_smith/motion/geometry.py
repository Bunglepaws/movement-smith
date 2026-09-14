from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Parent indices live in retarget.joints; we keep a copy here so geometry stays independent.
SMPLH_PARENTS_DEFAULT = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
    21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert HY-Motion / Zhou 6D rotations to matrices. `rot6d` is (..., 6).

    HY-Motion's `rot6d_to_rotation_matrix` views the 6 numbers as a (3, 2)
    block (first two matrix columns, row-major) then Gram-Schmidt. Concatenating
    `rot6d[..., :3]` / `[..., 3:6]` as columns is a different layout and makes
    hips/knees flip ~180° from one frame to the next.
    """
    arr = np.asarray(rot6d, dtype=np.float64)
    x = arr.reshape(*arr.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = _safe_normalize(a1)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = _safe_normalize(a2 - dot * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def fix_quat_continuity(quats: np.ndarray) -> np.ndarray:
    """Pick q or -q so consecutive frames have a positive dot product."""
    out = np.asarray(quats, dtype=np.float64).copy()
    if out.shape[0] <= 1:
        return out
    for t in range(1, out.shape[0]):
        flip = np.sum(out[t] * out[t - 1], axis=-1) < 0
        out[t][flip] *= -1
    return out


def geodesic_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Shortest angle in radians between wxyz quaternions. Broadcasts on the last dim."""
    dots = np.abs(np.sum(q0 * q1, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Slerp wxyz quats. `frac` broadcasts as (...,)."""
    dots = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dots < 0.0, -q1, q1)
    dots = np.abs(dots)
    frac = np.asarray(frac, dtype=np.float64)
    if frac.ndim < dots.ndim:
        frac = frac.reshape(frac.shape + (1,) * (dots.ndim - frac.ndim))
    linear = dots > 0.9995
    omega = np.arccos(np.clip(dots, 0.0, 1.0))
    sin_omega = np.sin(omega)
    a = np.sin((1.0 - frac) * omega) / np.maximum(sin_omega, 1e-8)
    b = np.sin(frac * omega) / np.maximum(sin_omega, 1e-8)
    out = a * q0 + b * q1
    lerp = q0 + frac * (q1 - q0)
    out = np.where(linear, lerp, out)
    return _safe_normalize(out)


def stabilize_rotations(quats: np.ndarray, max_step_rad: float = np.deg2rad(15.0)) -> np.ndarray:
    """Remove 180° flips and cap per-frame joint speed so limbs cannot kick."""
    q = fix_quat_continuity(quats)
    t_count = q.shape[0]
    if t_count < 2:
        return q
    if t_count >= 3:
        d_prev = geodesic_angle(q[1:-1], q[:-2])
        d_next = geodesic_angle(q[1:-1], q[2:])
        d_span = geodesic_angle(q[:-2], q[2:])
        spike = (d_prev > max_step_rad) & (d_next > max_step_rad) & (d_span < max_step_rad * 2.0)
        mid = _slerp_wxyz(q[:-2], q[2:], 0.5)
        q[1:-1] = np.where(spike[..., None], mid, q[1:-1])
        q = fix_quat_continuity(q)
    for t in range(1, t_count):
        ang = geodesic_angle(q[t - 1], q[t])
        frac = np.minimum(1.0, max_step_rad / np.maximum(ang, 1e-8))
        q[t] = _slerp_wxyz(q[t - 1], q[t], frac)
    return fix_quat_continuity(q)


def matrix_to_quat_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to (..., 4) wxyz quaternions."""
    shape = matrices.shape[:-2]
    flat = matrices.reshape(-1, 3, 3)
    quat_xyzw = Rotation.from_matrix(flat).as_quat()
    wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
    return wxyz.reshape(*shape, 4)


def axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle to (..., 4) wxyz quaternions."""
    shape = axis_angle.shape[:-1]
    flat = axis_angle.reshape(-1, 3)
    quat_xyzw = Rotation.from_rotvec(flat).as_quat()
    wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
    return wxyz.reshape(*shape, 4)


def quat_wxyz_to_matrix(quats: np.ndarray) -> np.ndarray:
    """Convert (..., 4) wxyz quaternions to (..., 3, 3) matrices."""
    shape = quats.shape[:-1]
    flat = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    xyzw = np.concatenate([flat[:, 1:4], flat[:, 0:1]], axis=1)
    mats = Rotation.from_quat(xyzw).as_matrix()
    return mats.reshape(*shape, 3, 3)


def identity_quats(n_frames: int, n_joints: int) -> np.ndarray:
    q = np.zeros((n_frames, n_joints, 4), dtype=np.float64)
    q[..., 0] = 1.0
    return q


def pad_or_trim_joints(quats: np.ndarray, n_joints: int = 52) -> np.ndarray:
    """Pad missing joints with identity or trim extras to SMPL-H size."""
    t, j, _ = quats.shape
    if j == n_joints:
        return quats
    if j > n_joints:
        return quats[:, :n_joints]
    out = identity_quats(t, n_joints)
    out[:, :j] = quats
    return out


def reshape_poses_axis_angle(poses: np.ndarray) -> np.ndarray:
    """Normalize poses to (T, J, 3) axis-angle."""
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 2:
        if arr.shape[1] % 3 != 0:
            raise ValueError(f"flat poses last dim must be divisible by 3, got {arr.shape}")
        return arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)
    raise ValueError(f"unsupported poses shape {arr.shape}")


def reshape_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Normalize rot6d to (T, J, 6). Drops a leading batch dim of 1."""
    arr = np.asarray(rot6d, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 6:
        return arr
    if arr.ndim == 2:
        if arr.shape[1] % 6 != 0:
            raise ValueError(f"flat rot6d last dim must be divisible by 6, got {arr.shape}")
        return arr.reshape(arr.shape[0], arr.shape[1] // 6, 6)
    raise ValueError(f"unsupported rot6d shape {arr.shape}")


def fk_world_rotations(local_quats: np.ndarray, parents: list[int] | None = None) -> np.ndarray:
    """Forward-kinematics world rotations from parent-relative wxyz quats.

    `local_quats` is (T, J, 4). Rest local rotation is identity (SMPL T-pose).
    """
    parents = parents if parents is not None else SMPLH_PARENTS_DEFAULT
    local_mats = quat_wxyz_to_matrix(local_quats)
    t, j, _, _ = local_mats.shape
    world = np.zeros_like(local_mats)
    for i, parent in enumerate(parents[:j]):
        if parent < 0:
            world[:, i] = local_mats[:, i]
        else:
            world[:, i] = world[:, parent] @ local_mats[:, i]
    return world


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


