from __future__ import annotations

from movement_smith.retarget.snapshot import SkeletonSnapshot
from pydantic import BaseModel, Field


class MapRequest(BaseModel):
    snapshot: SkeletonSnapshot
    override: dict[str, str] | None = None


class GenerateJobRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    duration_s: float = Field(default=4.0, ge=0.5, le=12.0)
    seed: int = Field(default=42, ge=0)
    cfg_scale: float = Field(default=5.0, ge=1.0, le=15.0)
    snapshot: SkeletonSnapshot
    override: dict[str, str] | None = None
    zero_root_xz: bool = False
