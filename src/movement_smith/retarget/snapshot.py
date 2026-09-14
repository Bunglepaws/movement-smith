from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class BoneSnapshot(BaseModel):
    name: str
    parent: str | None = None
    rest_local: list[float] = Field(min_length=16, max_length=16)
    rest_world: list[float] = Field(min_length=16, max_length=16)

    def local_matrix(self) -> np.ndarray:
        return _mat4_colmajor(self.rest_local)

    def world_matrix(self) -> np.ndarray:
        return _mat4_colmajor(self.rest_world)


class SkeletonSnapshot(BaseModel):
    bones: list[BoneSnapshot]

    def names(self) -> list[str]:
        return [bone.name for bone in self.bones]

    def by_name(self) -> dict[str, BoneSnapshot]:
        return {bone.name: bone for bone in self.bones}


def _mat4_colmajor(elements: list[float]) -> np.ndarray:
    return np.asarray(elements, dtype=np.float64).reshape(4, 4, order="F")


def rotation_from_matrix(mat: np.ndarray) -> np.ndarray:
    """Extract an orthonormal 3x3 rotation from a 4x4 (or 3x3) matrix."""
    r = np.asarray(mat, dtype=np.float64)[:3, :3]
    u, _, vt = np.linalg.svd(r)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def translation_from_matrix(mat: np.ndarray) -> np.ndarray:
    return np.asarray(mat, dtype=np.float64)[:3, 3].copy()
