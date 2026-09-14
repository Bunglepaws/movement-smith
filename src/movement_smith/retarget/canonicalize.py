"""Normalize bone names so Mixamo, VRM, UE, DAZ, and Rigify share one lookup key."""

from __future__ import annotations

import re

_PREFIX_RE = re.compile(
    r"^(?:"
    r"mixamorig:|mixamorig|mixamo:|mixamo|"
    r"j_bip_c_|j_bip_l_|j_bip_r_|j_bip_|"
    r"cc_base_|def_|bip_|armature\|armature_|"
    r"genesis[0-9]*_|"
    r"character1_|"
    r")"
    r"",
    re.IGNORECASE,
)
_SEP_RE = re.compile(r"[_\-:\s.]+")
# Blender often uses upper_armL / thighR with no separator before the side letter.
_BLENDER_SIDE_RE = re.compile(r"^(.+?)([LR])$")


def normalize_bone_name(name: str) -> str:
    """Lowercase, strip common rig prefixes, drop separators."""
    trimmed = name.strip()
    previous = None
    while previous != trimmed:
        previous = trimmed
        trimmed = _PREFIX_RE.sub("", trimmed)
    compacted = _SEP_RE.sub("", trimmed)
    return compacted.lower()


def side_and_core(name: str) -> tuple[str | None, str]:
    """Return (L|R|None, core) after prefix strip, before dropping separators."""
    trimmed = name.strip()
    previous = None
    while previous != trimmed:
        previous = trimmed
        trimmed = _PREFIX_RE.sub("", trimmed)
    lower = trimmed.lower()
    side: str | None = None
    if lower.startswith("left") or re.match(r"^l[_\-.]", lower) or lower.endswith(".l") or lower.endswith("_l"):
        side = "L"
    elif lower.startswith("right") or re.match(r"^r[_\-.]", lower) or lower.endswith(".r") or lower.endswith("_r"):
        side = "R"
    else:
        match = _BLENDER_SIDE_RE.match(trimmed)
        if match and match.group(1)[-1:].isalpha():
            side = match.group(2)
            trimmed = match.group(1)
    core = normalize_bone_name(trimmed)
    core = re.sub(r"^(left|right|l|r)", "", core)
    core = re.sub(r"(left|right|l|r)$", "", core)
    return side, core
