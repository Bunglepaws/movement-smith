"""Map uploaded skeleton bones onto SMPL-H joints."""

from __future__ import annotations

from dataclasses import dataclass, field

from movement_smith.retarget.canonicalize import normalize_bone_name, side_and_core
from movement_smith.retarget.joints import (
    BODY_JOINT_NAMES,
    HAND_JOINT_NAMES,
    SMPLH_JOINT_NAMES,
)

# Keys are normalize_bone_name() results.
_ALIAS: dict[str, str] = {}

_CENTER_JOINTS = {"Pelvis", "Spine1", "Spine2", "Spine3", "Neck", "Head"}


def _alias(names: list[str], joint: str) -> None:
    for name in names:
        _ALIAS[normalize_bone_name(name)] = joint


# We deliberately omit bare "Root"/"root": many Blender exports use root as an
# armature null above the hips, and mapping Pelvis there crumples the mesh.
_alias(["Pelvis", "Hips", "Hip", "pelvis", "hips", "hip"], "Pelvis")
_alias(["Spine", "Spine1", "spine_01", "abdomen", "abdomenLower"], "Spine1")
_alias(["Spine2", "spine_02", "abdomen2", "abdomenUpper", "Chest"], "Spine2")
_alias(["Spine3", "spine_03", "chest", "upperchest", "SpineUpper"], "Spine3")
_alias(["Neck", "neck_01", "neck"], "Neck")
_alias(["Head", "head"], "Head")

for side, smpl_side in (("Left", "L"), ("Right", "R"), ("l", "L"), ("r", "R")):
    _alias(
        [
            f"{side}Shoulder",
            f"{side}shoulder",
            f"clavicle_{side[0].lower()}",
            f"{smpl_side}_Collar",
            f"{smpl_side}Collar",
            f"{side}collar",
            f"shoulder.{side[0].lower()}",
        ],
        f"{smpl_side}_Collar",
    )
    _alias(
        [
            f"{side}Arm",
            f"{side}UpperArm",
            f"upperarm_{side[0].lower()}",
            f"{smpl_side}_Shoulder",
            f"{smpl_side}Shldr",
            f"{side}shldr",
            f"upper_arm.{side[0].lower()}",
        ],
        f"{smpl_side}_Shoulder",
    )
    _alias(
        [
            f"{side}ForeArm",
            f"{side}LowerArm",
            f"lowerarm_{side[0].lower()}",
            f"{smpl_side}_Elbow",
            f"{smpl_side}ForeArm",
            f"forearm.{side[0].lower()}",
        ],
        f"{smpl_side}_Elbow",
    )
    _alias(
        [
            f"{side}Hand",
            f"{side}Wrist",
            f"hand_{side[0].lower()}",
            f"{smpl_side}_Wrist",
            f"{smpl_side}Hand",
            f"hand.{side[0].lower()}",
        ],
        f"{smpl_side}_Wrist",
    )
    _alias(
        [
            f"{side}UpLeg",
            f"{side}UpperLeg",
            f"{side}Thigh",
            f"thigh_{side[0].lower()}",
            f"{smpl_side}_Hip",
            f"{smpl_side}Thigh",
            f"thigh.{side[0].lower()}",
        ],
        f"{smpl_side}_Hip",
    )
    _alias(
        [
            f"{side}Leg",
            f"{side}Shin",
            f"{side}Calf",
            f"calf_{side[0].lower()}",
            f"{smpl_side}_Knee",
            f"{smpl_side}Shin",
            f"shin.{side[0].lower()}",
        ],
        f"{smpl_side}_Knee",
    )
    _alias(
        [
            f"{side}Foot",
            f"{side}Ankle",
            f"foot_{side[0].lower()}",
            f"{smpl_side}_Ankle",
            f"{smpl_side}Foot",
            f"foot.{side[0].lower()}",
        ],
        f"{smpl_side}_Ankle",
    )
    _alias(
        [
            f"{side}ToeBase",
            f"{side}Toe",
            f"ball_{side[0].lower()}",
            f"{smpl_side}_Foot",
            f"{smpl_side}Toe",
            f"toe.{side[0].lower()}",
        ],
        f"{smpl_side}_Foot",
    )

# VRM 0.x / 1.0
_alias(["J_Bip_C_Hips"], "Pelvis")
_alias(["J_Bip_C_Spine"], "Spine1")
_alias(["J_Bip_C_Chest"], "Spine2")
_alias(["J_Bip_C_UpperChest"], "Spine3")
_alias(["J_Bip_C_Neck"], "Neck")
_alias(["J_Bip_C_Head"], "Head")
_alias(["J_Bip_L_Shoulder"], "L_Collar")
_alias(["J_Bip_R_Shoulder"], "R_Collar")
_alias(["J_Bip_L_UpperArm"], "L_Shoulder")
_alias(["J_Bip_R_UpperArm"], "R_Shoulder")
_alias(["J_Bip_L_LowerArm"], "L_Elbow")
_alias(["J_Bip_R_LowerArm"], "R_Elbow")
_alias(["J_Bip_L_Hand"], "L_Wrist")
_alias(["J_Bip_R_Hand"], "R_Wrist")
_alias(["J_Bip_L_UpperLeg"], "L_Hip")
_alias(["J_Bip_R_UpperLeg"], "R_Hip")
_alias(["J_Bip_L_LowerLeg"], "L_Knee")
_alias(["J_Bip_R_LowerLeg"], "R_Knee")
_alias(["J_Bip_L_Foot"], "L_Ankle")
_alias(["J_Bip_R_Foot"], "R_Ankle")
_alias(["J_Bip_L_ToeBase"], "L_Foot")
_alias(["J_Bip_R_ToeBase"], "R_Foot")

_CORE_TO_JOINT = {
    "hips": "Pelvis",
    "hip": "Pelvis",
    "pelvis": "Pelvis",
    "spine": "Spine1",
    "spine1": "Spine1",
    "spine01": "Spine1",
    "spine001": "Spine1",
    "abdomen": "Spine1",
    "spine2": "Spine2",
    "spine02": "Spine2",
    "spine002": "Spine2",
    "chest": "Spine2",
    "spine3": "Spine3",
    "spine03": "Spine3",
    "spine003": "Spine3",
    "upperchest": "Spine3",
    "neck": "Neck",
    "head": "Head",
    "shoulder": "Collar",
    "collar": "Collar",
    "clavicle": "Collar",
    "arm": "Shoulder",
    "upperarm": "Shoulder",
    "shldr": "Shoulder",
    "forearm": "Elbow",
    "lowerarm": "Elbow",
    "elbow": "Elbow",
    "hand": "Wrist",
    "wrist": "Wrist",
    "upleg": "Hip",
    "upperleg": "Hip",
    "thigh": "Hip",
    "leg": "Knee",
    "shin": "Knee",
    "calf": "Knee",
    "knee": "Knee",
    "foot": "Ankle",
    "ankle": "Ankle",
    "toebase": "Foot",
    "toe": "Foot",
    "ball": "Foot",
}


@dataclass
class BonePair:
    source: str
    target: str
    method: str


@dataclass
class MappingResult:
    pairs: list[BonePair]
    unmapped_sources: list[str]
    extra_targets: list[str]
    body_coverage: float
    hand_coverage: float
    coverage: float
    warning: str | None = None
    source_to_target: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "joint_order": list(SMPLH_JOINT_NAMES),
            "pairs": [
                {"source": p.source, "target": p.target, "method": p.method} for p in self.pairs
            ],
            "unmapped_sources": self.unmapped_sources,
            "extra_targets": self.extra_targets,
            "body_coverage": self.body_coverage,
            "hand_coverage": self.hand_coverage,
            "coverage": self.coverage,
            "warning": self.warning,
            "source_to_target": self.source_to_target,
        }


def map_skeleton(
    bone_names: list[str],
    override: dict[str, str] | None = None,
    coverage_warn: float = 0.5,
) -> MappingResult:
    """Assign each SMPL-H joint at most one target bone.

    Extra bones stay unmapped (rest pose). Override maps SMPL-H name -> target bone.
    """
    override = override or {}
    used_targets: set[str] = set()
    pairs: list[BonePair] = []
    assigned: dict[str, str] = {}
    mixamo_spine = _has_mixamo_spine_chain(bone_names)
    blender_spine = _has_blender_spine_chain(bone_names)

    for source, target in override.items():
        if source not in SMPLH_JOINT_NAMES:
            continue
        if target not in bone_names:
            continue
        pairs.append(BonePair(source=source, target=target, method="override"))
        assigned[source] = target
        used_targets.add(target)

    remaining_bones = [name for name in bone_names if name not in used_targets]
    for bone in remaining_bones:
        joint = _lookup_alias(bone, mixamo_spine=mixamo_spine, blender_spine=blender_spine)
        if joint is None or joint in assigned:
            continue
        pairs.append(BonePair(source=joint, target=bone, method="alias"))
        assigned[joint] = bone
        used_targets.add(bone)

    remaining_bones = [name for name in bone_names if name not in used_targets]
    remaining_sources = [j for j in SMPLH_JOINT_NAMES if j not in assigned]
    for bone in remaining_bones:
        joint = _fuzzy_joint(bone, remaining_sources)
        if joint is None:
            continue
        pairs.append(BonePair(source=joint, target=bone, method="fuzzy"))
        assigned[joint] = bone
        used_targets.add(bone)
        remaining_sources.remove(joint)

    _assign_root_pelvis_fallback(bone_names, assigned, pairs, used_targets)

    unmapped = [j for j in SMPLH_JOINT_NAMES if j not in assigned]
    extra = [name for name in bone_names if name not in used_targets]
    body_hit = sum(1 for j in BODY_JOINT_NAMES if j in assigned)
    hand_hit = sum(1 for j in HAND_JOINT_NAMES if j in assigned)
    body_coverage = body_hit / len(BODY_JOINT_NAMES)
    hand_coverage = hand_hit / len(HAND_JOINT_NAMES) if HAND_JOINT_NAMES else 0.0
    coverage = len(assigned) / len(SMPLH_JOINT_NAMES)
    warning = None
    if body_coverage < coverage_warn:
        warning = f"body coverage {body_coverage:.0%} is below {coverage_warn:.0%}"
    return MappingResult(
        pairs=pairs,
        unmapped_sources=unmapped,
        extra_targets=extra,
        body_coverage=body_coverage,
        hand_coverage=hand_coverage,
        coverage=coverage,
        warning=warning,
        source_to_target=assigned,
    )


def _has_mixamo_spine_chain(bone_names: list[str]) -> bool:
    keys = {normalize_bone_name(name) for name in bone_names}
    return "spine" in keys and "spine1" in keys


def _has_blender_spine_chain(bone_names: list[str]) -> bool:
    """Blender often parents thighs under bare `spine` with spine001/002/003 above."""
    keys = {normalize_bone_name(name) for name in bone_names}
    return "spine" in keys and bool(keys & {"spine001", "spine01"})


def _lookup_alias(bone: str, *, mixamo_spine: bool, blender_spine: bool) -> str | None:
    key = normalize_bone_name(bone)
    if blender_spine and key in {"spine", "spine001", "spine01", "spine002", "spine02", "spine003", "spine03"}:
        return {
            "spine": "Pelvis",
            "spine001": "Spine1",
            "spine01": "Spine1",
            "spine002": "Spine2",
            "spine02": "Spine2",
            "spine003": "Spine3",
            "spine03": "Spine3",
        }[key]
    if not mixamo_spine and key in {"spine1", "spine2", "spine3"}:
        return {"spine1": "Spine1", "spine2": "Spine2", "spine3": "Spine3"}[key]
    if mixamo_spine and key in {"spine", "spine1", "spine2"}:
        return {"spine": "Spine1", "spine1": "Spine2", "spine2": "Spine3"}[key]
    if key in _ALIAS:
        return _ALIAS[key]
    side, core = side_and_core(bone)
    core_joint = _CORE_TO_JOINT.get(core)
    if core_joint is None:
        return None
    if core_joint in _CENTER_JOINTS:
        # Sided names like pelvisL are control bones, not the SMPL pelvis.
        if side is not None:
            return None
        return core_joint
    if side is None:
        return None
    return f"{side}_{core_joint}"


def _assign_root_pelvis_fallback(
    bone_names: list[str],
    assigned: dict[str, str],
    pairs: list[BonePair],
    used_targets: set[str],
) -> None:
    """Only claim bare root as Pelvis when nothing better was found."""
    if "Pelvis" in assigned:
        return
    for bone in bone_names:
        if normalize_bone_name(bone) != "root":
            continue
        if bone in used_targets:
            return
        pairs.append(BonePair(source="Pelvis", target=bone, method="root-fallback"))
        assigned["Pelvis"] = bone
        used_targets.add(bone)
        return


def _fuzzy_joint(bone: str, candidates: list[str], min_ratio: float = 0.78) -> str | None:
    key = normalize_bone_name(bone)
    best: tuple[float, str] | None = None
    for joint in candidates:
        if not _fuzzy_allowed(key, joint):
            continue
        ratio = _similarity(key, normalize_bone_name(joint))
        if best is None or ratio > best[0]:
            best = (ratio, joint)
    if best is None or best[0] < min_ratio:
        return None
    return best[1]


def _fuzzy_allowed(bone_key: str, joint: str) -> bool:
    """Block anatomically silly fuzzy matches (spine ↔ pelvis, etc.)."""
    if bone_key.startswith("spine") and joint == "Pelvis":
        return False
    if bone_key in {"pelvis", "hips", "hip", "root"} and joint.startswith("Spine"):
        return False
    if bone_key.startswith("toe") and joint in {"L_Ankle", "R_Ankle"}:
        return False
    if bone_key in {"pelvisl", "pelvisr"} and joint == "Pelvis":
        return False
    return True


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = _levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]
