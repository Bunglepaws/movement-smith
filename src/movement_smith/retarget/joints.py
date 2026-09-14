"""SMPL-H joint names, parents, and body vs hand split."""

from __future__ import annotations

SMPLH_JOINT_NAMES: list[str] = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

SMPLH_PARENTS: list[int] = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    21,
    37,
    38,
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
]

SMPLH_JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(SMPLH_JOINT_NAMES)}

BODY_JOINT_NAMES: list[str] = SMPLH_JOINT_NAMES[:22]
HAND_JOINT_NAMES: list[str] = SMPLH_JOINT_NAMES[22:]
N_BODY_JOINTS = 22
N_HAND_JOINTS = 30

# Approximate T-pose pelvis height in metres for SMPL-H neutral.
SMPLH_HIP_HEIGHT_M = 0.92

# Clavicles fold cartoon arms into the torso. Wrists / ankles / toes are noisy on
# stylized binds; we leave those bones at rest.
SKIP_BY_DEFAULT: frozenset[str] = frozenset(
    {
        "L_Collar",
        "R_Collar",
        "L_Wrist",
        "R_Wrist",
        "L_Ankle",
        "R_Ankle",
        "L_Foot",
        "R_Foot",
    }
)

