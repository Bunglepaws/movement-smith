import numpy as np

from movement_smith.motion.convert import clip_from_rot6d
from movement_smith.motion.geometry import (
    geodesic_angle,
    rot6d_to_matrix,
    stabilize_rotations,
)


def test_hymotion_identity_6d_is_identity_matrix() -> None:
    """HY-Motion stores [R00, R01, R10, R11, R20, R21], not concatenated columns."""
    rot6d = np.zeros((3, 22, 6))
    rot6d[..., 0] = 1.0
    rot6d[..., 3] = 1.0
    mats = rot6d_to_matrix(rot6d)
    assert np.allclose(mats, np.eye(3), atol=1e-6)


def test_hymotion_6d_uses_view_3x2_columns() -> None:
    rng = np.random.default_rng(0)
    r6 = rng.normal(size=(8, 6))
    x = r6.reshape(8, 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    expected = np.stack([b1, b2, b3], axis=-1)
    assert np.allclose(rot6d_to_matrix(r6), expected)


def test_clip_from_rot6d_identity_keeps_unit_quats() -> None:
    t = 4
    rot6d = np.zeros((t, 22, 6))
    rot6d[..., 0] = 1.0
    rot6d[..., 3] = 1.0
    trans = np.zeros((t, 3))
    clip = clip_from_rot6d(rot6d, trans)
    assert clip.rotations_numpy().shape == (t, 52, 4)
    assert np.allclose(clip.rotations_numpy()[:, :22, 0], 1.0, atol=1e-5)


def test_stabilize_caps_single_frame_kicks() -> None:
    q = np.zeros((5, 1, 4))
    q[..., 0] = 1.0
    # 150° jump on X at frame 2, then back.
    q[2, 0] = [np.cos(np.deg2rad(75)), np.sin(np.deg2rad(75)), 0.0, 0.0]
    out = stabilize_rotations(q, max_step_rad=np.deg2rad(15.0))
    jumps = geodesic_angle(out[1:], out[:-1]) * 180.0 / np.pi
    assert float(jumps.max()) < 16.0