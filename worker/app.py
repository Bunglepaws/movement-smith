from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException

from movement_smith.motion.schema import MotionClip
from worker.models import GenerateRequest
from worker.runtime import WorkerRuntime, load_runtime

_runtime: WorkerRuntime | None = None
_runtime_error: str | None = None


def get_runtime() -> WorkerRuntime:
    global _runtime, _runtime_error
    if _runtime is not None:
        return _runtime
    if _runtime_error:
        raise HTTPException(status_code=503, detail=_runtime_error)
    try:
        _runtime = load_runtime()
    except Exception as exc:
        _runtime_error = str(exc)
        raise HTTPException(status_code=503, detail=_runtime_error) from exc
    return _runtime


def set_runtime(runtime: WorkerRuntime | None) -> None:
    global _runtime, _runtime_error
    _runtime = runtime
    _runtime_error = None


def verify_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("MOVEMENT_SMITH_WORKER_TOKEN", "").strip()
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid worker token")


def create_app(runtime: WorkerRuntime | None = None) -> FastAPI:
    if runtime is not None:
        set_runtime(runtime)

    app = FastAPI(title="movement-smith-worker", version="0.1.0")

    @app.get("/v1/health")
    def health(_: None = Depends(verify_token)) -> dict:
        variant = os.environ.get("HYMOTION_VARIANT", "lite")
        try:
            rt = get_runtime()
        except HTTPException as exc:
            return {
                "ok": False,
                "model_id": "hy-motion-1.0",
                "variant": variant,
                "detail": exc.detail,
            }
        return {
            "ok": True,
            "model_id": rt.model_id,
            "variant": rt.variant,
            "detail": None,
        }

    @app.post("/v1/motions")
    def generate(req: GenerateRequest, _: None = Depends(verify_token)) -> MotionClip:
        rt = get_runtime()
        return rt.generate(
            prompt=req.prompt,
            duration_s=req.duration_s,
            seed=req.seed,
            cfg_scale=req.cfg_scale,
        )

    return app


app = create_app()
