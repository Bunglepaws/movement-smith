from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

SKELETON_SMPLH: Literal["smplh"] = "smplh"
DEFAULT_FPS = 30.0


class MotionClip(BaseModel):
    """Canonical clip produced by a text-to-motion worker.

    Rotations are parent-relative quaternions in wxyz order. Root translation
    is in metres, Y-up. Joint order matches `joint_names` (SMPL-H, 52 joints).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skeleton: Literal["smplh"] = SKELETON_SMPLH
    fps: float = DEFAULT_FPS
    joint_names: list[str]
    root_trans: list[list[float]] = Field(description="(T, 3) pelvis translation")
    rotations_quat: list[list[list[float]]] = Field(
        description="(T, J, 4) wxyz parent-relative rotations"
    )
    prompt: str | None = None
    seed: int | None = None
    model_id: str | None = None

    @property
    def n_frames(self) -> int:
        return len(self.root_trans)

    @property
    def duration_s(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.n_frames / self.fps

    def trans_numpy(self) -> np.ndarray:
        return np.asarray(self.root_trans, dtype=np.float64)

    def rotations_numpy(self) -> np.ndarray:
        return np.asarray(self.rotations_quat, dtype=np.float64)

    @classmethod
    def from_numpy(
        cls,
        *,
        joint_names: list[str],
        root_trans: np.ndarray,
        rotations_quat: np.ndarray,
        fps: float = DEFAULT_FPS,
        prompt: str | None = None,
        seed: int | None = None,
        model_id: str | None = None,
    ) -> MotionClip:
        trans = np.asarray(root_trans, dtype=np.float64)
        quats = np.asarray(rotations_quat, dtype=np.float64)
        if trans.ndim != 2 or trans.shape[1] != 3:
            raise ValueError(f"root_trans must be (T, 3), got {trans.shape}")
        if quats.ndim != 3 or quats.shape[2] != 4:
            raise ValueError(f"rotations_quat must be (T, J, 4), got {quats.shape}")
        if trans.shape[0] != quats.shape[0]:
            raise ValueError("root_trans and rotations_quat frame counts differ")
        if quats.shape[1] != len(joint_names):
            raise ValueError("joint_names length must match rotations_quat J")
        return cls(
            skeleton=SKELETON_SMPLH,
            fps=fps,
            joint_names=list(joint_names),
            root_trans=trans.tolist(),
            rotations_quat=quats.tolist(),
            prompt=prompt,
            seed=seed,
            model_id=model_id,
        )
