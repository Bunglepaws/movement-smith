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
