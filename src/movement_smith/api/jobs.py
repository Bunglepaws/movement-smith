from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from movement_smith.motion.provider import MotionProvider
from movement_smith.motion.schema import MotionClip
from movement_smith.retarget.mapping import map_skeleton
from movement_smith.retarget.retarget import RetargetedClip, retarget_clip
from movement_smith.retarget.snapshot import SkeletonSnapshot

JobStatus = Literal["pending", "running", "done", "error"]


class JobStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.jobs_dir = data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._running = False

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    async def enqueue(
        self,
        *,
        prompt: str,
        duration_s: float,
        seed: int,
        cfg_scale: float,
        snapshot: SkeletonSnapshot,
        override: dict[str, str] | None,
        zero_root_xz: bool,
        provider: MotionProvider,
    ) -> dict[str, Any]:
        with self._lock:
            if self._running:
                raise RuntimeError("a generation job is already running")
            job_id = uuid.uuid4().hex[:12]
            mapping = map_skeleton(snapshot.names(), override=override)
            record: dict[str, Any] = {
                "id": job_id,
                "status": "pending",
                "prompt": prompt,
                "duration_s": duration_s,
                "seed": seed,
                "cfg_scale": cfg_scale,
                "mapping": mapping.as_dict(),
                "error": None,
                "clip": None,
                "source": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._jobs[job_id] = record
            self._running = True
            self._write(record)

        thread = threading.Thread(
            target=lambda: asyncio.run(
                self._run(
                    job_id=job_id,
                    prompt=prompt,
                    duration_s=duration_s,
                    seed=seed,
                    cfg_scale=cfg_scale,
                    snapshot=snapshot,
                    mapping=mapping,
                    zero_root_xz=zero_root_xz,
                    provider=provider,
                )
            ),
            daemon=True,
        )
        thread.start()
        return record

    async def _run(
        self,
        *,
        job_id: str,
        prompt: str,
        duration_s: float,
        seed: int,
        cfg_scale: float,
        snapshot: SkeletonSnapshot,
        mapping: Any,
        zero_root_xz: bool,
        provider: MotionProvider,
    ) -> None:
        record = self._jobs[job_id]
        record["status"] = "running"
        self._write(record)
        try:
            motion: MotionClip = await provider.generate(
                prompt=prompt,
                duration_s=duration_s,
                seed=seed,
                cfg_scale=cfg_scale,
            )
            retargeted: RetargetedClip = retarget_clip(
                motion,
                snapshot,
                mapping,
                zero_root_xz=zero_root_xz,
            )
            record["clip"] = retargeted.model_dump()
            record["source"] = motion.model_dump()
            record["motion_meta"] = {
                "model_id": motion.model_id,
                "fps": motion.fps,
                "n_frames": motion.n_frames,
                "duration_s": motion.duration_s,
            }
            record["status"] = "done"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        finally:
            with self._lock:
                self._running = False
            self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        path = self.jobs_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
