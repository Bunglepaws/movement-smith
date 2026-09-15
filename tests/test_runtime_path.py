from pathlib import Path

import pytest

from worker.runtime import resolve_device_ids, resolve_model_path


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


def test_device_ids_require_cuda() -> None:
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device_ids(
            False,
            None,
            executable="/venv/bin/python",
            torch_version="2.5.1+cpu",
            torch_cuda=None,
            cuda_visible_devices=None,
        )


def test_device_ids_default_to_gpu_zero() -> None:
    assert resolve_device_ids(
        True,
        None,
        executable="/venv/bin/python",
        torch_version="2.5.1+cu124",
        torch_cuda="12.4",
        cuda_visible_devices=None,
    ) == [0]


def test_device_ids_parse_env() -> None:
    assert resolve_device_ids(
        True,
        "0,1",
        executable="/venv/bin/python",
        torch_version="2.5.1+cu124",
        torch_cuda="12.4",
        cuda_visible_devices="0,1",
    ) == [0, 1]
