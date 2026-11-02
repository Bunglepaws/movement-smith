from movement_smith.retarget.joints import BODY_JOINT_NAMES
from movement_smith.retarget.mapping import map_skeleton

MIXAMO_BONES = [
    "mixamorig:Hips",
    "mixamorig:Spine",
    "mixamorig:Spine1",
    "mixamorig:Spine2",
    "mixamorig:Neck",
    "mixamorig:Head",
    "mixamorig:LeftShoulder",
    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",
    "mixamorig:LeftHand",
    "mixamorig:RightShoulder",
    "mixamorig:RightArm",
    "mixamorig:RightForeArm",
    "mixamorig:RightHand",
    "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg",
    "mixamorig:LeftFoot",
    "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg",
    "mixamorig:RightLeg",
    "mixamorig:RightFoot",
    "mixamorig:RightToeBase",
    "mixamorig:LeftArmTwist",
    "mixamorig:RightArmTwist",
]


def test_mixamo_maps_body_and_leaves_twist_extra() -> None:
    result = map_skeleton(MIXAMO_BONES)
    assert result.body_coverage == 1.0
    extras = set(result.extra_targets)
    assert "mixamorig:LeftArmTwist" in extras
    assert "mixamorig:RightArmTwist" in extras
    assert result.source_to_target["Pelvis"] == "mixamorig:Hips"
    assert result.source_to_target["L_Shoulder"] == "mixamorig:LeftArm"
    assert result.source_to_target["L_Collar"] == "mixamorig:LeftShoulder"
    assert result.source_to_target["Spine1"] == "mixamorig:Spine"
    assert result.source_to_target["Spine2"] == "mixamorig:Spine1"


def test_missing_clavicles_lowers_coverage_without_failing() -> None:
    bones = [b for b in MIXAMO_BONES if "Shoulder" not in b and "Twist" not in b]
    result = map_skeleton(bones)
    assert "L_Collar" in result.unmapped_sources
    assert "R_Collar" in result.unmapped_sources
    assert result.body_coverage >= 0.8
    assert result.warning is None


def test_smpl_named_spine_is_not_shifted() -> None:
    bones = ["Pelvis", "Spine1", "Spine2", "Spine3", "Neck", "Head"]
    result = map_skeleton(bones)
    assert result.source_to_target["Spine1"] == "Spine1"
    assert result.source_to_target["Spine2"] == "Spine2"
    assert result.source_to_target["Spine3"] == "Spine3"


def test_override_wins() -> None:
    result = map_skeleton(
        MIXAMO_BONES,
        override={"Pelvis": "mixamorig:Spine"},
    )
    assert result.source_to_target["Pelvis"] == "mixamorig:Spine"


def test_low_coverage_warns() -> None:
    result = map_skeleton(["Hips", "Head"])
    assert result.body_coverage < 0.5
    assert result.warning is not None
    assert len(BODY_JOINT_NAMES) == 22


def test_blender_spine_chain_maps_pelvis_to_bare_spine() -> None:
    bones = [
        "root",
        "pelvisL",
        "spine",
        "spine001",
        "spine002",
        "spine003",
        "head",
        "shoulderL",
        "upper_armL",
        "forearmL",
        "handL",
        "shoulderR",
        "upper_armR",
        "forearmR",
        "handR",
        "thighL",
        "shinL",
        "footL",
        "toeL",
        "thighR",
        "shinR",
        "footR",
        "toeR",
    ]
    result = map_skeleton(bones)
    assert result.source_to_target["Pelvis"] == "spine"
    assert result.source_to_target["Spine1"] == "spine001"
    assert result.source_to_target["Spine2"] == "spine002"
    assert result.source_to_target["Spine3"] == "spine003"
    assert result.source_to_target["L_Hip"] == "thighL"
    assert result.source_to_target["L_Collar"] == "shoulderL"
    assert "root" in result.extra_targets
    assert "pelvisL" in result.extra_targets


def test_rigify_def_hyphen_prefix_maps_body() -> None:
    bones = [
        "DEF-spine",
        "DEF-spine001",
        "DEF-spine002",
        "DEF-spine003",
        "DEF-thighL",
        "DEF-shinL",
        "DEF-footL",
        "DEF-toeL",
        "DEF-thighR",
        "DEF-shinR",
        "DEF-footR",
        "DEF-toeR",
        "DEF-shoulderL",
        "DEF-upper_armL",
        "DEF-forearmL",
        "DEF-handL",
        "DEF-shoulderR",
        "DEF-upper_armR",
        "DEF-forearmR",
        "DEF-handR",
    ]
    result = map_skeleton(bones)
    assert result.body_coverage >= 0.9
    assert result.source_to_target["Pelvis"] == "DEF-spine"
    assert result.source_to_target["L_Hip"] == "DEF-thighL"


def test_hip_rooted_blender_spine_does_not_map_spine_to_pelvis() -> None:
    bones = [
        "hips",
        "pelvis.L",
        "pelvis.R",
        "spine",
        "spine.001",
        "spine.002",
        "neck",
        "head",
        "shoulder.L",
        "upper_arm.L",
        "forearm.L",
        "hand.L",
        "shoulder.R",
        "upper_arm.R",
        "forearm.R",
        "hand.R",
        "thigh.L",
        "shin.L",
        "foot.L",
        "toe.L",
        "thigh.R",
        "shin.R",
        "foot.R",
        "toe.R",
        "f_index.01.L",
        "f_index.02.L",
        "f_index.03.L",
        "f_index.01.R",
        "f_index.02.R",
        "f_index.03.R",
    ]
    result = map_skeleton(bones)
    assert result.source_to_target["Pelvis"] == "hips"
    assert result.source_to_target["Spine1"] == "spine"
    assert result.source_to_target["Spine2"] == "spine.001"
    assert result.source_to_target["Spine3"] == "spine.002"
    assert result.source_to_target["L_Index1"] == "f_index.01.L"
    assert result.source_to_target["R_Index3"] == "f_index.03.R"
    assert "pelvis.L" in result.extra_targets
    assert result.body_coverage == 1.0


def test_four_finger_glued_rigify_hands_map() -> None:
    """Exporters turn f_index.01.L into f_index01L; some binds omit the ring."""
    bones = [
        "hips",
        "pelvisL",
        "pelvisR",
        "spine",
        "spine001",
        "spine002",
        "neck",
        "head",
        "shoulderL",
        "upper_armL",
        "forearmL",
        "handL",
        "shoulderR",
        "upper_armR",
        "forearmR",
        "handR",
        "thighL",
        "shinL",
        "footL",
        "toeL",
        "thighR",
        "shinR",
        "footR",
        "toeR",
        "palm01L",
        "f_index01L",
        "f_index02L",
        "f_index03L",
        "thumb01L",
        "thumb02L",
        "thumb03L",
        "palm02L",
        "f_middle01L",
        "f_middle02L",
        "palm04L",
        "f_pinky01L",
        "f_pinky02L",
        "f_pinky03L",
        "palm01R",
        "f_index01R",
        "f_index02R",
        "f_index03R",
        "thumb01R",
        "thumb02R",
        "thumb03R",
        "palm02R",
        "f_middle01R",
        "f_middle02R",
        "palm04R",
        "f_pinky01R",
        "f_pinky02R",
        "f_pinky03R",
    ]
    result = map_skeleton(bones)
    assert result.body_coverage == 1.0
    assert result.hand_coverage > 0.7
    assert result.source_to_target["L_Wrist"] == "handL"
    assert result.source_to_target["L_Index1"] == "f_index01L"
    assert result.source_to_target["L_Index3"] == "f_index03L"
    assert result.source_to_target["L_Thumb2"] == "thumb02L"
    assert result.source_to_target["L_Middle1"] == "f_middle01L"
    assert result.source_to_target["L_Middle2"] == "f_middle02L"
    assert result.source_to_target["L_Pinky3"] == "f_pinky03L"
    assert result.source_to_target["R_Index1"] == "f_index01R"
    assert "L_Ring1" not in result.source_to_target
    assert "L_Middle3" not in result.source_to_target
    assert "palm01L" in result.extra_targets
    assert "palm04R" in result.extra_targets

