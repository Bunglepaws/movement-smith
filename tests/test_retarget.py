import numpy as np

from movement_smith.motion.geometry import identity_quats
from movement_smith.motion.schema import MotionClip
from movement_smith.retarget.joints import SMPLH_JOINT_INDEX, SMPLH_JOINT_NAMES
from movement_smith.retarget.mapping import map_skeleton
from movement_smith.retarget.retarget import retarget_clip
from movement_smith.retarget.snapshot import BoneSnapshot, SkeletonSnapshot


def _identity_matrix(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        tx,
        ty,
        tz,
        1.0,
    ]


def _snapshot(names: list[str], hip: str = "Hips") -> SkeletonSnapshot:
    bones = []
    parent = None
    for i, name in enumerate(names):
        ty = 0.92 if name == hip else 0.1 * i
        bones.append(
            BoneSnapshot(
                name=name,
                parent=parent if i else None,
                rest_local=_identity_matrix(ty=0.1 if i else 0.0),
                rest_world=_identity_matrix(ty=ty),
            )
        )
        parent = name
    return SkeletonSnapshot(bones=bones)


def test_retarget_writes_tracks_for_mapped_bones_only() -> None:
    names = [
        "Hips",
        "Spine",
        "LeftArm",
        "LeftArmTwist",
        "LeftForeArm",
        "RightArm",
        "RightForeArm",
        "LeftUpLeg",
        "LeftLeg",
        "RightUpLeg",
        "RightLeg",
        "Head",
        "Neck",
        "LeftShoulder",
        "RightShoulder",
        "LeftHand",
        "RightHand",
        "LeftFoot",
        "RightFoot",
        "LeftToeBase",
        "RightToeBase",
        "Spine1",
        "Spine2",
    ]
    snapshot = _snapshot(names)
    mapping = map_skeleton(names)
    n_frames = 10
    quats = identity_quats(n_frames, len(SMPLH_JOINT_NAMES))
    trans = np.zeros((n_frames, 3))
    trans[:, 1] = 0.92
    trans[:, 0] = np.linspace(0, 1, n_frames)
    clip = MotionClip.from_numpy(
        joint_names=list(SMPLH_JOINT_NAMES),
        root_trans=trans,
        rotations_quat=quats,
        fps=30,
    )
    out = retarget_clip(clip, snapshot, mapping, zero_root_xz=True)
    tracked = {t.bone for t in out.tracks}
    assert "LeftArmTwist" not in tracked
    assert "LeftShoulder" not in tracked
    assert "RightShoulder" not in tracked
    assert "Hips" in tracked
    hips = next(t for t in out.tracks if t.bone == "Hips")
    # In-place skips position tracks so we never write world-ish coords into local .position.
    assert hips.position is None

    moving = retarget_clip(clip, snapshot, mapping, zero_root_xz=False)
    hips_m = next(t for t in moving.tracks if t.bone == "Hips")
    assert hips_m.position is not None
    assert hips_m.position[0][0] != hips_m.position[-1][0]


def test_identity_motion_preserves_non_identity_rest() -> None:
    """Identity SMPL should leave each bone at its rest local rotation."""
    rest = [
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.0,
        1.0,
    ]
    snapshot = SkeletonSnapshot(
        bones=[
            BoneSnapshot(
                name="Hips",
                parent=None,
                rest_local=_identity_matrix(ty=0.92),
                rest_world=_identity_matrix(ty=0.92),
            ),
            BoneSnapshot(
                name="Spine",
                parent="Hips",
                rest_local=rest,
                rest_world=rest,
            ),
        ]
    )
    mapping = map_skeleton(["Hips", "Spine"])
    quats = identity_quats(4, len(SMPLH_JOINT_NAMES))
    clip = MotionClip.from_numpy(
        joint_names=list(SMPLH_JOINT_NAMES),
        root_trans=np.zeros((4, 3)),
        rotations_quat=quats,
    )
    out = retarget_clip(clip, snapshot, mapping, zero_root_xz=True)
    spine = next(t for t in out.tracks if t.bone == "Spine")
    from movement_smith.motion.geometry import matrix_to_quat_wxyz
    from movement_smith.retarget.snapshot import rotation_from_matrix, _mat4_colmajor

    expected = matrix_to_quat_wxyz(rotation_from_matrix(_mat4_colmajor(rest))[None, ...])[0]
    got = np.array(spine.rotation[0])
    if np.dot(got, expected) < 0:
        got = -got
    assert np.allclose(got, expected, atol=1e-5)


def test_identity_accounts_for_armature_parent_world() -> None:
    """Bone world can include a non-bone Armature rotation that must not enter locals."""
    rx = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.92,
        0.0,
        1.0,
    ]
    snapshot = SkeletonSnapshot(
        bones=[
            BoneSnapshot(
                name="Hips",
                parent=None,
                rest_local=_identity_matrix(ty=0.92),
                rest_world=rx,
            )
        ]
    )
    mapping = map_skeleton(["Hips"])
    quats = identity_quats(4, len(SMPLH_JOINT_NAMES))
    clip = MotionClip.from_numpy(
        joint_names=list(SMPLH_JOINT_NAMES),
        root_trans=np.zeros((4, 3)),
        rotations_quat=quats,
    )
    out = retarget_clip(clip, snapshot, mapping, zero_root_xz=True)
    hips = next(t for t in out.tracks if t.bone == "Hips")
    got = np.array(hips.rotation[0])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    if np.dot(got, ident) < 0:
        got = -got
    assert np.allclose(got, ident, atol=1e-5)


def test_inplace_freezes_pelvis_to_rest() -> None:
    names = ["Hips", "LeftUpLeg", "RightUpLeg"]
    snapshot = _snapshot(names)
    mapping = map_skeleton(names)
    n_frames = 8
    quats = identity_quats(n_frames, len(SMPLH_JOINT_NAMES))
    # Spin the SMPL pelvis hard; in-place should ignore it.
    from movement_smith.retarget.joints import SMPLH_JOINT_INDEX

    t = np.linspace(0, np.pi, n_frames)
    quats[:, SMPLH_JOINT_INDEX["Pelvis"], 0] = np.cos(t / 2)
    quats[:, SMPLH_JOINT_INDEX["Pelvis"], 2] = np.sin(t / 2)
    clip = MotionClip.from_numpy(
        joint_names=list(SMPLH_JOINT_NAMES),
        root_trans=np.zeros((n_frames, 3)),
        rotations_quat=quats,
    )
    out = retarget_clip(clip, snapshot, mapping, zero_root_xz=True)
    hips = next(tr for tr in out.tracks if tr.bone == "Hips")
    q0 = np.array(hips.rotation[0])
    for q in hips.rotation[1:]:
        got = np.array(q)
        if np.dot(got, q0) < 0:
            got = -got
        assert np.allclose(got, q0, atol=1e-5)

    names = ["Hips", "Head"]
    snapshot = _snapshot(names)
    mapping = map_skeleton(names)
    quats = identity_quats(4, len(SMPLH_JOINT_NAMES))
    trans = np.zeros((4, 3))
    clip = MotionClip.from_numpy(
        joint_names=list(SMPLH_JOINT_NAMES),
        root_trans=trans,
        rotations_quat=quats,
    )
    out = retarget_clip(clip, snapshot, mapping)
    tracked = {t.bone for t in out.tracks}
    assert tracked <= {"Hips", "Head"}
    assert "LeftArm" not in tracked


def test_skipped_collar_does_not_double_apply_spine() -> None:
    """Clavicles stay at rest local; spine rotation must not hit the arm twice."""
    names = ["Hips", "Spine", "LeftShoulder", "LeftArm"]
    snapshot = _snapshot(names)
    mapping = map_skeleton(names)
    assert mapping.source_to_target["L_Collar"] == "LeftShoulder"
    assert mapping.source_to_target["L_Shoulder"] == "LeftArm"
    n_frames = 4
    quats = identity_quats(n_frames, len(SMPLH_JOINT_NAMES))
    quats[:, SMPLH_JOINT_INDEX["Spine1"], 0] = np.cos(np.pi / 4)
    quats[:, SMPLH_JOINT_INDEX["Spine1"], 2] = np.sin(np.pi / 4)
    clip = MotionClip.from_numpy(
        joint_names=list(SMPLH_JOINT_NAMES),
        root_trans=np.zeros((n_frames, 3)),
        rotations_quat=quats,
    )
    out = retarget_clip(clip, snapshot, mapping, zero_root_xz=True)
    tracked = {t.bone for t in out.tracks}
    assert "LeftShoulder" not in tracked
    arm = next(t for t in out.tracks if t.bone == "LeftArm")
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    for q in arm.rotation:
        got = np.array(q)
        if np.dot(got, ident) < 0:
            got = -got
        assert np.allclose(got, ident, atol=1e-5)
