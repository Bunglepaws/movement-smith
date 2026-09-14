from __future__ import annotations

import numpy as np

from movement_smith.motion.geometry import (
    axis_angle_to_quat_wxyz,
    matrix_to_quat_wxyz,
    pad_or_trim_joints,
    reshape_poses_axis_angle,
    reshape_rot6d,
    rot6d_to_matrix,
    stabilize_rotations,
)
from movement_smith.motion.schema import DEFAULT_FPS, MotionClip
from movement_smith.retarget.joints import SMPLH_JOINT_NAMES


def clip_from_smplh_poses(
    poses: np.ndarray,
    trans: np.ndarray,
    *,
    fps: float = DEFAULT_FPS,
    prompt: str | None = None,
    seed: int | None = None,
    model_id: str | None = None,
) -> MotionClip:
    """Build a MotionClip from SMPL-H axis-angle poses and root translation."""
    aa = reshape_poses_axis_angle(poses)
    trans_arr = _as_trans(trans)
    if trans_arr.shape[0] != aa.shape[0]:
        raise ValueError("poses and trans frame counts differ")
    quats = pad_or_trim_joints(axis_angle_to_quat_wxyz(aa), n_joints=len(SMPLH_JOINT_NAMES))
    quats = stabilize_rotations(quats)
    return MotionClip.from_numpy(
        joint_names=list(SMPLH_JOINT_NAMES),
        root_trans=trans_arr,
        rotations_quat=quats,
        fps=fps,
        prompt=prompt,
        seed=seed,
        model_id=model_id,
    )


def clip_from_rot6d(
    rot6d: np.ndarray,
    transl: np.ndarray,
    *,
    fps: float = DEFAULT_FPS,
    prompt: str | None = None,
    seed: int | None = None,
    model_id: str | None = None,
) -> MotionClip:
    """Build a MotionClip from HY-Motion rot6d + transl tensors."""
    r6 = reshape_rot6d(rot6d)
    trans_arr = _as_trans(transl)
    if trans_arr.shape[0] != r6.shape[0]:
        raise ValueError("rot6d and transl frame counts differ")
    mats = rot6d_to_matrix(r6)
    quats = pad_or_trim_joints(matrix_to_quat_wxyz(mats), n_joints=len(SMPLH_JOINT_NAMES))
    quats = stabilize_rotations(quats)
    return MotionClip.from_numpy(
        joint_names=list(SMPLH_JOINT_NAMES),
        root_trans=trans_arr,
        rotations_quat=quats,
        fps=fps,
        prompt=prompt,
        seed=seed,
        model_id=model_id,
    )


def clip_from_hymotion_output(
    model_output: dict,
    *,
    fps: float = DEFAULT_FPS,
    prompt: str | None = None,
    seed: int | None = None,
    model_id: str = "hy-motion-1.0",
    sample_index: int = 0,
) -> MotionClip:
    """Convert T2MRuntime.generate_motion model_output (or NPZ-like dict) to MotionClip."""
    if "rot6d" in model_output and "transl" in model_output:
        rot6d = _maybe_torch_to_numpy(model_output["rot6d"])
        transl = _maybe_torch_to_numpy(model_output["transl"])
        rot6d = _index_batch(rot6d, sample_index)
        transl = _index_batch(transl, sample_index)
        return clip_from_rot6d(
            rot6d, transl, fps=fps, prompt=prompt, seed=seed, model_id=model_id
        )
    if "poses" in model_output and "trans" in model_output:
        poses = _maybe_torch_to_numpy(model_output["poses"])
        trans = _maybe_torch_to_numpy(model_output["trans"])
        return clip_from_smplh_poses(
            poses, trans, fps=fps, prompt=prompt, seed=seed, model_id=model_id
        )
    raise ValueError("model_output needs rot6d+transl or poses+trans")


def _as_trans(trans: np.ndarray) -> np.ndarray:
    arr = np.asarray(_maybe_torch_to_numpy(trans), dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"trans must be (T, 3), got {arr.shape}")
    return arr


def _index_batch(arr: np.ndarray, index: int) -> np.ndarray:
    """Drop a leading sample/batch dim without treating time as batch."""
    if arr.ndim == 4:
        return arr[index]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _maybe_torch_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
