from __future__ import annotations

from urllib.parse import urlparse

import httpx

from movement_smith.motion.provider import ProviderStatus
from movement_smith.motion.schema import MotionClip


class RemoteHttpProvider:
    """POST /v1/motions on a self-hosted or Modal worker."""

    id = "remote"

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        modal_key: str | None = None,
        modal_secret: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.modal_key = modal_key
        self.modal_secret = modal_secret
        self.timeout_s = timeout_s

    def host_label(self) -> str:
        parsed = urlparse(self.base_url)
        return parsed.netloc or self.base_url

    def is_modal(self) -> bool:
        """True when the worker URL or proxy-auth tokens point at Modal."""
        host = (urlparse(self.base_url).hostname or "").lower()
        if host.endswith(".modal.run") or host.endswith(".modal.local"):
            return True
        return bool(self.modal_key and self.modal_secret)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.modal_key:
            headers["Modal-Key"] = self.modal_key
        if self.modal_secret:
            headers["Modal-Secret"] = self.modal_secret
        return headers

    async def generate(
        self,
        prompt: str,
        duration_s: float,
        seed: int,
        cfg_scale: float = 5.0,
    ) -> MotionClip:
        payload = {
            "prompt": prompt,
            "duration_s": duration_s,
            "seed": seed,
            "cfg_scale": cfg_scale,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/v1/motions",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            return MotionClip.model_validate(response.json())

    async def health(self) -> ProviderStatus:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/health",
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            return ProviderStatus(
                ok=False,
                model_id="hy-motion-1.0",
                detail=str(exc),
            )
        return ProviderStatus(
            ok=bool(data.get("ok", False)),
            model_id=str(data.get("model_id", "hy-motion-1.0")),
            variant=data.get("variant"),
            detail=data.get("detail"),
        )
