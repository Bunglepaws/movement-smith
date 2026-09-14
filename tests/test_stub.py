import asyncio

import numpy as np

from movement_smith.motion.convert import clip_from_rot6d, clip_from_smplh_poses
from movement_smith.motion.schema import MotionClip
from movement_smith.motion.stub import StubProvider
from movement_smith.retarget.joints import SMPLH_JOINT_NAMES


def test_stub_clip_shape() -> None:
    provider = StubProvider()
    clip = asyncio.run(provider.generate("A person walks forward", duration_s=2.0, seed=1))
    assert isinstance(clip, MotionClip)
    assert clip.n_frames == 60
    assert clip.rotations_numpy().shape == (60, len(SMPLH_JOINT_NAMES), 4)
    norms = np.linalg.norm(clip.rotations_numpy(), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_clip_from_poses_flat() -> None:
    t = 5
    poses = np.zeros((t, 156))
    trans = np.zeros((t, 3))
    clip = clip_from_smplh_poses(poses, trans)
    assert clip.rotations_numpy().shape[1] == 52


def test_clip_from_rot6d_pads_body_only() -> None:
    t = 4
    rot6d = np.zeros((t, 22, 6))
    rot6d[..., 0] = 1.0
    rot6d[..., 3] = 1.0
    trans = np.zeros((t, 3))
    clip = clip_from_rot6d(rot6d, trans)
    assert clip.rotations_numpy().shape == (t, 52, 4)
