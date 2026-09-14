from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from movement_smith.motion.schema import MotionClip


class ProviderStatus(BaseModel):
    ok: bool
    model_id: str
    variant: str | None = None
    detail: str | None = None


@runtime_checkable
class MotionProvider(Protocol):
    id: str

    async def generate(
        self,
        prompt: str,
        duration_s: float,
        seed: int,
        cfg_scale: float = 5.0,
    ) -> MotionClip: ...

    async def health(self) -> ProviderStatus: ...
