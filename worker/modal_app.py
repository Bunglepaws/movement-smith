"""Optional Modal deploy of the movement-smith HY-Motion worker.

The local API talks to this process the same way it talks to a LAN GPU:
set MOVEMENT_SMITH_WORKER_URL to the printed HTTPS URL and set
MOVEMENT_SMITH_MODAL_KEY / MOVEMENT_SMITH_MODAL_SECRET for proxy auth.

Commands:
  modal serve worker/modal_app.py
  modal deploy worker/modal_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

app = modal.App("movement-smith-worker")

weights = modal.Volume.from_name("movement-smith-hymotion-weights", create_if_missing=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
HYMOTION_GIT = "https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git"

# Local sources are mounted last (no copy=True). PYTHONPATH imports them at
# container start so local edits do not force a full image rebuild.
image = (
  modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel").apt_install("git", "git-lfs").run_commands(
    "git lfs install",
    f"git clone --depth 1 {HYMOTION_GIT} /opt/HY-Motion-1.0",
    # WoodenMesh bins are LFS objects; shallow clone can leave pointer stubs.
    "cd /opt/HY-Motion-1.0 && git lfs pull",
    "test -s /opt/HY-Motion-1.0/scripts/gradio/static/assets/dump_wooden/v_template.bin",
    "pip install -r /opt/HY-Motion-1.0/requirements.txt",
    "pip install fastapi uvicorn pydantic pydantic-settings numpy scipy httpx huggingface_hub",
  ).env({
    "HYMOTION_ROOT": "/opt/HY-Motion-1.0",
    "HYMOTION_VARIANT": os.environ.get("HYMOTION_VARIANT", "lite"),
    "DISABLE_PROMPT_ENGINEERING": "True",
    # HY-Motion text_encoder.py reads this at import time.
    "USE_HF_MODELS": "1",
    # Persist CLIP / Qwen downloads across cold starts.
    "HF_HOME": "/weights/hf_home",
    "PYTHONPATH": "/opt/movement-smith:/opt/movement-smith/src:/opt/HY-Motion-1.0",
  }).add_local_dir(str(REPO_ROOT / "src"), remote_path="/opt/movement-smith/src").add_local_dir(str(REPO_ROOT / "worker"), remote_path="/opt/movement-smith/worker")
)


def _download_weights(variant: str) -> str:
  from huggingface_hub import snapshot_download

  token = os.environ.get("HF_TOKEN")
  repo_id = "tencent/HY-Motion-1.0"
  dest = Path("/weights") / variant
  dest.mkdir(parents=True, exist_ok=True)
  allow = ["HY-Motion-1.0-Lite/**"] if variant == "lite" else ["HY-Motion-1.0/**"]
  snapshot_download(
    repo_id=repo_id,
    local_dir=str(dest),
    allow_patterns=list(allow),
    token=token,
  )
  # Warm HF hub cache for text encoders (USE_HF_MODELS=1).
  Path("/weights/hf_home").mkdir(parents=True, exist_ok=True)
  snapshot_download("openai/clip-vit-large-patch14", token=token)
  snapshot_download("Qwen/Qwen3-8B", token=token)
  if variant == "lite":
    return str(dest / "HY-Motion-1.0-Lite")
  return str(dest / "HY-Motion-1.0")


@app.cls(
  image=image,
  gpu="A100-40GB",
  timeout=600,
  startup_timeout=3600,
  scaledown_window=60,
  max_containers=1,
  volumes={"/weights": weights},
  secrets=[modal.Secret.from_name("huggingface")],
)
class HymotionWorker:

  @modal.enter()
  def setup(self) -> None:
    # Must be set before any hymotion import (text_encoder reads at import time).
    os.environ["USE_HF_MODELS"] = "1"
    os.environ["HF_HOME"] = "/weights/hf_home"
    os.environ["DISABLE_PROMPT_ENGINEERING"] = "True"
    os.environ["HYMOTION_ROOT"] = "/opt/HY-Motion-1.0"

    sys.path.insert(0, "/opt/movement-smith")
    sys.path.insert(0, "/opt/movement-smith/src")
    sys.path.insert(0, "/opt/HY-Motion-1.0")
    os.chdir("/opt/HY-Motion-1.0")

    variant = os.environ.get("HYMOTION_VARIANT", "lite")
    model_path = _download_weights(variant)
    os.environ["HYMOTION_MODEL_PATH"] = model_path
    weights.commit()
    from worker.runtime import load_runtime

    self.runtime = load_runtime()

  @modal.asgi_app(requires_proxy_auth=True)
  def api(self):
    from worker.app import create_app

    return create_app(self.runtime)
