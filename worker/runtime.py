"""Load official HY-Motion T2MRuntime from HYMOTION_ROOT."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from movement_smith.motion.convert import clip_from_hymotion_output
from movement_smith.motion.schema import DEFAULT_FPS, MotionClip


_VARIANT_DIRS = {
    "lite": "HY-Motion-1.0-Lite",
    "full": "HY-Motion-1.0",
}


def resolve_model_path(root: Path, variant: str, model_path_env: str | None) -> Path:
    if variant not in _VARIANT_DIRS:
        raise ValueError("HYMOTION_VARIANT must be lite or full")
    default = root / "ckpts" / "tencent" / _VARIANT_DIRS[variant]
    raw = (model_path_env or "").strip()
    if not raw:
        return default
    model_path = Path(raw).expanduser()
    implied = next((name for name, dirname in _VARIANT_DIRS.items() if dirname == model_path.name), None)
    if implied is not None and implied != variant:
        raise ValueError(
            f"HYMOTION_VARIANT={variant} but HYMOTION_MODEL_PATH is {model_path} "
            f"(that directory is the {implied} checkpoint). Unset HYMOTION_MODEL_PATH "
            f"to use {default}, or point it at the {variant} weights."
        )
    return model_path


@dataclass
class WorkerRuntime:
    t2m: object
    variant: str
    model_id: str = "hy-motion-1.0"

    def generate(self, prompt: str, duration_s: float, seed: int, cfg_scale: float) -> MotionClip:
        html, fbx_files, model_output = self.t2m.generate_motion(
            text=prompt,
            seeds_csv=str(seed),
            duration=duration_s,
            cfg_scale=cfg_scale,
            output_format="dict",
            original_text=prompt,
        )
        del html, fbx_files
        return clip_from_hymotion_output(
            model_output,
            fps=DEFAULT_FPS,
            prompt=prompt,
            seed=seed,
            model_id=self.model_id,
            sample_index=0,
        )


def load_runtime() -> WorkerRuntime:
    root = Path(os.environ.get("HYMOTION_ROOT", "")).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            "HYMOTION_ROOT is not a directory. Clone Tencent-Hunyuan/HY-Motion-1.0 and point HYMOTION_ROOT at it."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    variant = os.environ.get("HYMOTION_VARIANT", "lite").strip().lower()
    model_path = resolve_model_path(root, variant, os.environ.get("HYMOTION_MODEL_PATH"))
    cfg = model_path / "config.yml"
    ckpt = model_path / "latest.ckpt"
    if not cfg.is_file() or not ckpt.is_file():
        raise FileNotFoundError(
            f"HY-Motion weights missing at {model_path} (need config.yml and latest.ckpt)"
        )

    from hymotion.utils.t2m_runtime import T2MRuntime

    # WoodenMesh opens scripts/gradio/static/assets/dump_wooden relative to cwd.
    wooden = root / "scripts" / "gradio" / "static" / "assets" / "dump_wooden" / "v_template.bin"
    if not wooden.is_file():
        raise FileNotFoundError(
            f"HY-Motion wooden mesh assets missing at {wooden.parent}. "
            "Clone with git-lfs and run `git lfs pull` inside HYMOTION_ROOT."
        )
    os.chdir(root)

    disable_pe = os.environ.get("DISABLE_PROMPT_ENGINEERING", "True").lower() in {
        "1",
        "true",
        "yes",
    }
    t2m = T2MRuntime(
        config_path=str(cfg),
        ckpt_name=str(ckpt),
        disable_prompt_engineering=disable_pe,
    )
    return WorkerRuntime(t2m=t2m, variant=variant)
