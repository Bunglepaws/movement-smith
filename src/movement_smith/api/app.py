from __future__ import annotations

import time
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from movement_smith.api.jobs import JobStore
from movement_smith.api.models import GenerateJobRequest, MapRequest
from movement_smith.config import Settings, get_settings
from movement_smith.motion.provider import MotionProvider
from movement_smith.motion.remote import RemoteHttpProvider
from movement_smith.motion.stub import StubProvider
from movement_smith.retarget.mapping import map_skeleton


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore(settings.data_dir)
    provider = _build_provider(settings)
    health_cache: dict = {"t": 0.0, "body": None}

    app = FastAPI(title="movement-smith", version="0.1.0")
    origins = [item.strip() for item in settings.cors_origins.split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        worker_host = None
        if isinstance(provider, RemoteHttpProvider):
            worker_host = provider.host_label()
        # Modal cold-starts on any authenticated request. We report config only
        # so the studio chip never wakes a GPU container.
        if isinstance(provider, RemoteHttpProvider) and provider.is_modal():
            return {
                "ok": True,
                "provider": provider.id,
                "stub": False,
                "worker_ok": True,
                "worker_host": worker_host,
                "model_id": "hy-motion-1.0",
                "variant": None,
                "detail": "Modal worker; not probed until Generate",
                "probe_worker": False,
                "modal": True,
            }
        now = time.monotonic()
        cached = health_cache["body"]
        if cached is not None and now - health_cache["t"] < 8.0:
            return cached
        status = await provider.health()
        payload = {
            "ok": True,
            "provider": provider.id,
            "stub": provider.id == "stub",
            "worker_ok": status.ok,
            "worker_host": worker_host,
            "model_id": status.model_id,
            "variant": status.variant,
            "detail": status.detail,
            "probe_worker": True,
            "modal": False,
        }
        health_cache["t"] = now
        health_cache["body"] = payload
        return payload

    @app.post("/api/skeletons/map")
    async def map_bones(req: MapRequest) -> dict:
        result = map_skeleton(req.snapshot.names(), override=req.override)
        return result.as_dict()

    @app.post("/api/jobs")
    async def create_job(req: GenerateJobRequest) -> dict:
        if not req.snapshot.bones:
            raise HTTPException(status_code=400, detail="snapshot has no bones")
        try:
            record = await store.enqueue(
                prompt=req.prompt,
                duration_s=req.duration_s,
                seed=req.seed,
                cfg_scale=req.cfg_scale,
                snapshot=req.snapshot,
                override=req.override,
                zero_root_xz=req.zero_root_xz,
                provider=provider,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _public_job(record)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _public_job(record)

    return app


def _build_provider(settings: Settings) -> MotionProvider:
    if settings.worker_url.strip():
        parsed = urlparse(settings.worker_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("MOVEMENT_SMITH_WORKER_URL must be an http(s) URL")
        return RemoteHttpProvider(
            settings.worker_url,
            token=settings.worker_token or None,
            modal_key=settings.modal_key or None,
            modal_secret=settings.modal_secret or None,
        )
    if settings.allow_stub:
        return StubProvider()
    return RemoteHttpProvider("http://127.0.0.1:8100")


def _public_job(record: dict) -> dict:
    return {
        "id": record["id"],
        "status": record["status"],
        "prompt": record["prompt"],
        "duration_s": record["duration_s"],
        "seed": record["seed"],
        "mapping": record["mapping"],
        "error": record["error"],
        "clip": record["clip"],
        "source": record.get("source"),
        "motion_meta": record.get("motion_meta"),
        "created_at": record["created_at"],
    }


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("movement_smith.api.app:app", host="127.0.0.1", port=8765, reload=True)
