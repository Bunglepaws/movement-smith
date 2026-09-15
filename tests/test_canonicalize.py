from movement_smith.retarget.canonicalize import normalize_bone_name, side_and_core


def test_strips_mixamo_prefix() -> None:
    assert normalize_bone_name("mixamorig:LeftArm") == "leftarm"
    assert normalize_bone_name("mixamorigHips") == "hips"


def test_strips_vrm_prefix() -> None:
    assert normalize_bone_name("J_Bip_L_UpperArm") == "upperarm"


def test_strips_def_and_separators() -> None:
    assert normalize_bone_name("DEF_upper_arm.L") == "upperarml"
    assert normalize_bone_name("DEF-thighL") == "thighl"


def test_blender_side_suffix() -> None:
    assert side_and_core("upper_armL") == ("L", "upperarm")
    assert side_and_core("thighR") == ("R", "thigh")
    assert side_and_core("pelvisL") == ("L", "pelvis")
    assert side_and_core("f_index.01.L") == ("L", "findex01")
    assert side_and_core("thumb.02.R") == ("R", "thumb02")

