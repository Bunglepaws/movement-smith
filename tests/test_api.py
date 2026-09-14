from pathlib import Path
import time

from fastapi.testclient import TestClient

from movement_smith.api.app import create_app
from movement_smith.config import Settings


def _identity(tx: float = 0.0, ty: float = 0.92) -> list[float]:
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, 0, 1]


def test_modal_health_skips_remote_probe(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            allow_stub=False,
            data_dir=tmp_path,
            worker_url="https://example--worker-api.modal.run",
            modal_key="wk-test",
            modal_secret="ws-test",
        )
    )
    client = TestClient(app)
    health = client.get("/api/health").json()
    assert health["modal"] is True
    assert health["probe_worker"] is False
    assert health["worker_ok"] is True
    assert "modal.run" in (health["worker_host"] or "")


def test_health_and_stub_job(tmp_path: Path) -> None:
    app = create_app(Settings(allow_stub=True, data_dir=tmp_path, worker_url=""))
    client = TestClient(app)
    health = client.get("/api/health").json()
    assert health["stub"] is True
    assert health["provider"] == "stub"
    assert health.get("modal") is False

    snapshot = {
        "bones": [
            {
                "name": "Hips",
                "parent": None,
                "rest_local": _identity(),
                "rest_world": _identity(),
            },
            {
                "name": "LeftArm",
                "parent": "Hips",
                "rest_local": _identity(ty=0.1),
                "rest_world": _identity(ty=1.1),
            },
        ]
    }
    mapped = client.post("/api/skeletons/map", json={"snapshot": snapshot}).json()
    assert mapped["source_to_target"]["Pelvis"] == "Hips"

    created = client.post(
        "/api/jobs",
        json={
            "prompt": "A person waves",
            "duration_s": 1.0,
            "seed": 1,
            "snapshot": snapshot,
            "zero_root_xz": True,
        },
    )
    assert created.status_code == 200
    job_id = created.json()["id"]
    job = created.json()
    for _ in range(80):
        if job["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
        job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["clip"]["n_frames"] > 0
    assert job["source"]["rotations_quat"]
    bones = {t["bone"] for t in job["clip"]["tracks"]}
    assert "Hips" in bones
