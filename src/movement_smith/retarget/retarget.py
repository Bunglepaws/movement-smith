"""Retarget SMPL-H MotionClip rotations onto a snapshot skeleton."""

from __future__ import annotations

from pydantic import BaseModel

import numpy as np

from movement_smith.motion.geometry import matrix_to_quat_wxyz, quat_wxyz_to_matrix
from movement_smith.motion.schema import MotionClip
from movement_smith.retarget.joints import (
    HAND_JOINT_NAMES,
    SKIP_BY_DEFAULT,
    SMPLH_HIP_HEIGHT_M,
    SMPLH_JOINT_INDEX,
    SMPLH_JOINT_NAMES,
    SMPLH_PARENTS,
)
from movement_smith.retarget.mapping import MappingResult
from movement_smith.retarget.snapshot import (
    SkeletonSnapshot,
    rotation_from_matrix,
    translation_from_matrix,
)

_I3 = np.eye(3, dtype=np.float64)


class BoneTrack(BaseModel):
    bone: str
    rotation: list[list[float]]
    position: list[list[float]] | None = None


class RetargetedClip(BaseModel):
    fps: float
    n_frames: int
    quat_order: str = "wxyz"
    tracks: list[BoneTrack]
    hip_bone: str | None = None


def retarget_clip(
    clip: MotionClip,
    snapshot: SkeletonSnapshot,
    mapping: MappingResult,
    *,
    zero_root_xz: bool = False,
    apply_hands: bool | None = None,
) -> RetargetedClip:
    """Apply SMPL world rotations as deltas on the target bind pose.

    `R_world_anim = R_smpl_world @ R_rest_world`. Locals are recovered with the
    real parent world, including a non-bone Armature parent recovered as
    `R_world @ R_local^T`. Using bone-only FK here double-counts Blender's
    armature rotation and crumples the mesh.
    """
    source_to_target = dict(mapping.source_to_target)
    for joint in SKIP_BY_DEFAULT:
        source_to_target.pop(joint, None)
    if apply_hands is None:
        apply_hands = _both_hands_present(source_to_target)
    if not apply_hands:
        for joint in HAND_JOINT_NAMES:
            source_to_target.pop(joint, None)

    bones = snapshot.by_name()
    parents = {name: bone.parent if bone.parent in bones else None for name, bone in bones.items()}
    rest_local = {name: rotation_from_matrix(bone.local_matrix()) for name, bone in bones.items()}
    rest_world = {name: rotation_from_matrix(bone.world_matrix()) for name, bone in bones.items()}

    smpl_local = quat_wxyz_to_matrix(clip.rotations_numpy())
    if zero_root_xz:
        smpl_local[:, SMPLH_JOINT_INDEX["Pelvis"]] = _I3
    smpl_world = _fk_smpl(smpl_local)
    n_frames = clip.n_frames

    anim_world: dict[str, np.ndarray] = {}
    for name in bones:
        joint = next((j for j, t in source_to_target.items() if t == name), None)
        if joint is None:
            anim_world[name] = np.repeat(rest_world[name][None, ...], n_frames, axis=0)
        else:
            anim_world[name] = smpl_world[:, SMPLH_JOINT_INDEX[joint]] @ rest_world[name]

    hip_bone = source_to_target.get("Pelvis")
    hip_positions = None
    if not zero_root_xz and hip_bone is not None and hip_bone in bones:
        hip_positions = _hip_local_positions(clip.trans_numpy(), bones[hip_bone])

    tracks: list[BoneTrack] = []
    for joint, target in source_to_target.items():
        if target not in bones:
            continue
        parent = parents.get(target)
        if parent is not None:
            parent_world = anim_world[parent]
        else:
            implicit = rest_world[target] @ rest_local[target].T
            parent_world = np.repeat(implicit[None, ...], n_frames, axis=0)
        local = np.transpose(parent_world, (0, 2, 1)) @ anim_world[target]
        quats = matrix_to_quat_wxyz(local)
        _ensure_quat_continuity(quats)
        position = None
        if target == hip_bone and hip_positions is not None:
            position = hip_positions.tolist()
        tracks.append(
            BoneTrack(
                bone=target,
                rotation=quats.tolist(),
                position=position,
            )
        )

    return RetargetedClip(
        fps=clip.fps,
        n_frames=n_frames,
        tracks=tracks,
        hip_bone=hip_bone,
    )


def _fk_smpl(local_mats: np.ndarray) -> np.ndarray:
    t, j, _, _ = local_mats.shape
    world = np.zeros_like(local_mats)
    for i, parent in enumerate(SMPLH_PARENTS[:j]):
        if parent < 0:
            world[:, i] = local_mats[:, i]
        else:
            world[:, i] = world[:, parent] @ local_mats[:, i]
    return world


def _both_hands_present(assigned: dict[str, str]) -> bool:
    left = "L_Wrist" in assigned
    right = "R_Wrist" in assigned
    finger_hits = sum(1 for j in HAND_JOINT_NAMES if j in assigned)
    return left and right and finger_hits >= 10


def _hip_local_positions(trans: np.ndarray, hip_bone) -> np.ndarray:
    """Root translation as a delta on the hip bone's rest local translation."""
    rest_local = translation_from_matrix(hip_bone.local_matrix())
    rest_world = translation_from_matrix(hip_bone.world_matrix())
    hip_y = abs(float(rest_world[1]))
    scale = hip_y / SMPLH_HIP_HEIGHT_M if hip_y > 1e-3 else 1.0
    delta = (trans - trans[0]) * scale
    return rest_local[None, :] + delta


def _ensure_quat_continuity(quats: np.ndarray) -> None:
    """Flip quaternion signs so consecutive frames take the short arc."""
    for i in range(1, quats.shape[0]):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] *= -1


_ = SMPLH_JOINT_NAMES
