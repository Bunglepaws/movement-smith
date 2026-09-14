from pathlib import Path

import pytest

from worker.runtime import resolve_model_path


ROOT = Path("/data/HY-Motion-1.0")


def test_variant_selects_default_checkpoint_dir() -> None:
    assert resolve_model_path(ROOT, "lite", None) == ROOT / "ckpts/tencent/HY-Motion-1.0-Lite"
    assert resolve_model_path(ROOT, "full", None) == ROOT / "ckpts/tencent/HY-Motion-1.0"
    assert resolve_model_path(ROOT, "full", "") == ROOT / "ckpts/tencent/HY-Motion-1.0"


def test_custom_model_path_is_used_when_it_does_not_name_the_other_variant() -> None:
    custom = Path("/mnt/weights/hy-motion")
    assert resolve_model_path(ROOT, "full", str(custom)) == custom


def test_model_path_that_names_the_other_variant_is_rejected() -> None:
    lite = str(ROOT / "ckpts/tencent/HY-Motion-1.0-Lite")
    with pytest.raises(ValueError, match="HYMOTION_VARIANT=full"):
        resolve_model_path(ROOT, "full", lite)

    full = str(ROOT / "ckpts/tencent/HY-Motion-1.0")
    with pytest.raises(ValueError, match="HYMOTION_VARIANT=lite"):
        resolve_model_path(ROOT, "lite", full)


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="lite or full"):
        resolve_model_path(ROOT, "medium", None)
