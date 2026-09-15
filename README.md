# movement-smith

The local studio uploads a rigged `.glb` or `.fbx`, sends a text prompt to a GPU worker running [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0), retargets the SMPL-H clip onto the uploaded skeleton, and plays it in the browser. Extra bones stay at rest. Missing or renamed bones are skipped or fuzzy-matched.

HY-Motion generates humanoid motion only.

The Python import path is `movement_smith`. Environment variables use the `MOVEMENT_SMITH_` prefix.

## Layout

| Path | Role |
| --- | --- |
| `src/movement_smith/` | Local FastAPI app, motion contract, retargeting. No CUDA. |
| `frontend/` | Vite + React + Three.js studio. |
| `worker/` | HY-Motion HTTP worker. Same app on a LAN GPU or on Modal. |
| `tests/` | Mapping and retarget tests. No GPU. |

The local API calls `POST {MOVEMENT_SMITH_WORKER_URL}/v1/motions`. The host can be a LAN GPU or a Modal HTTPS URL. The API does not distinguish the two.

## Local studio

Python 3.11 or newer. From the repository root:

```
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env
```

`.env` defaults to stub motion (`MOVEMENT_SMITH_ALLOW_STUB=1`) so the UI runs without a GPU.

```
.venv/bin/python -m uvicorn movement_smith.api.app:app --host 127.0.0.1 --port 8765
```

Frontend:

```
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api` to port 8765.

Tests:

```
uv run pytest
```

## GPU worker

Clone [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) and download weights as described in that repository (`ckpts/README.md`). The Tencent Hunyuan Community License applies to those weights. The Hugging Face repo is gated.

Lite needs about 24 GB VRAM. Full needs about 26 GB. Prompt rewriting is off by default (`DISABLE_PROMPT_ENGINEERING=True`) so duration comes from the UI slider.

On the GPU host, from this repository, with HY-Motion on `PYTHONPATH` via `HYMOTION_ROOT`:

```
export HYMOTION_ROOT=/path/to/HY-Motion-1.0
export HYMOTION_VARIANT=lite
export DISABLE_PROMPT_ENGINEERING=True
export MOVEMENT_SMITH_WORKER_TOKEN=optional-shared-secret
. $HYMOTION_ROOT/.venv/bin/activate
cd /path/to/movement-smith
PYTHONPATH=. python -m uvicorn worker.app:app --host 0.0.0.0 --port 8100
```

The worker process must use the HY-Motion virtualenv, which has a CUDA build of PyTorch. `movement-smith/.venv` is CPU-only. If `torch.cuda.is_available()` is false, T2MRuntime reports `devices=cpu` and bitsandbytes offloads Qwen with a meta-device warning.

`HYMOTION_VARIANT` selects `ckpts/tencent/HY-Motion-1.0-Lite` or `ckpts/tencent/HY-Motion-1.0`. Set `HYMOTION_MODEL_PATH` only to override that directory. The worker rejects a path whose last component is the other variant's checkpoint name.

On the studio machine:

```
MOVEMENT_SMITH_ALLOW_STUB=0
MOVEMENT_SMITH_WORKER_URL=http://<gpu-host>:8100
MOVEMENT_SMITH_WORKER_TOKEN=optional-shared-secret
```

`GET /v1/health` reports whether the runtime loaded. `POST /v1/motions` returns a `MotionClip` (SMPL-H quaternions, 30 fps). A later model is a second worker that emits the same JSON.

## Modal worker (optional)

Modal is a fallback when no LAN GPU is available. It serves the same FastAPI app.

Create a Hugging Face token secret named `huggingface` (key `HF_TOKEN`) so the container can pull gated HY-Motion weights. Create Modal proxy-auth tokens in the Modal dashboard.

```
uv pip install "modal>=0.73"
modal serve worker/modal_app.py
```

`modal serve` prints a temporary HTTPS URL. `modal deploy worker/modal_app.py` prints a stable URL. First start downloads weights into the `movement-smith-hymotion-weights` Volume and loads the model. That can take several minutes. `scaledown_window` is 15 minutes so a studio session is not a cold start on every Generate.

Studio env:

```
MOVEMENT_SMITH_ALLOW_STUB=0
MOVEMENT_SMITH_WORKER_URL=https://<workspace>--movement-smith-worker-hymotionworker-api.modal.run
MOVEMENT_SMITH_MODAL_KEY=<proxy-token-id>
MOVEMENT_SMITH_MODAL_SECRET=<proxy-token-secret>
```

Default GPU is A100 40 GB. L4 24 GB can run Lite with short clips (`num_seeds=1`, duration under 5 s) if the class `gpu=` argument is changed.

The public URL requires Modal proxy auth. Do not leave an unauthenticated generator on the internet.

## Motion contract

Workers return:

- `skeleton`: `"smplh"`
- `fps`: 30
- `joint_names`: 52 SMPL-H names
- `root_trans`: `(T, 3)` metres, Y-up
- `rotations_quat`: `(T, J, 4)` wxyz, parent-relative

The local app retargets that clip onto the browser skeleton snapshot (bone names plus rest local/world matrices).

## Limits

- Humanoid motion only.
- Extra non-body bones do not animate.
- Seamless loop is not a HY-Motion feature.
- Download is animated GLB. FBX download is not in v1.
- Stub motion is used only when `MOVEMENT_SMITH_ALLOW_STUB=1` and `MOVEMENT_SMITH_WORKER_URL` is empty.
- Modal cold start after idle includes weight and model load.
- Studio `GET /api/health` does not call a Modal worker. It only reports that Modal is configured. The first Generate request wakes the container.

## Env

| Variable | Where | Meaning |
| --- | --- | --- |
| `MOVEMENT_SMITH_WORKER_URL` | studio | Worker base URL (LAN or Modal). |
| `MOVEMENT_SMITH_WORKER_TOKEN` | studio + worker | Optional Bearer token. |
| `MOVEMENT_SMITH_MODAL_KEY` | studio | Modal-Key header. |
| `MOVEMENT_SMITH_MODAL_SECRET` | studio | Modal-Secret header. |
| `MOVEMENT_SMITH_ALLOW_STUB` | studio | `1` to use synthetic motion when no worker URL is set. |
| `MOVEMENT_SMITH_DATA_DIR` | studio | Job JSON directory. Default `data/`. |
| `HYMOTION_ROOT` | worker | Clone of HY-Motion-1.0. |
| `HYMOTION_VARIANT` | worker | `lite` or `full`. Selects the default checkpoint directory under `HYMOTION_ROOT`. |
| `HYMOTION_MODEL_PATH` | worker | Optional. Directory with `config.yml` and `latest.ckpt`. Defaults from `HYMOTION_VARIANT`. |
| `HYMOTION_DEVICE_IDS` | worker | Optional. Comma-separated GPU ids. Default `0`. |
| `DISABLE_PROMPT_ENGINEERING` | worker | Default `True`. |
| `HF_TOKEN` | Modal secret | Hugging Face access for gated DiT weights (and encoder downloads). |
| `USE_HF_MODELS` | worker | Set to `1` on Modal so CLIP/Qwen load from Hugging Face IDs. Default in HY-Motion is `0` (local `ckpts/`). |
| `HF_HOME` | Modal worker | Cache dir for CLIP/Qwen; on Modal this is `/weights/hf_home` on the Volume. |
