from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    duration_s: float = Field(default=4.0, ge=0.5, le=12.0)
    seed: int = Field(default=42, ge=0)
    cfg_scale: float = Field(default=5.0, ge=1.0, le=15.0)
