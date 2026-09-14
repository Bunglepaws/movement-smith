from __future__ import annotations

import math

import numpy as np

from movement_smith.motion.geometry import identity_quats
from movement_smith.motion.provider import ProviderStatus
from movement_smith.motion.schema import DEFAULT_FPS, MotionClip
from movement_smith.retarget.joints import SMPLH_JOINT_INDEX, SMPLH_JOINT_NAMES


class StubProvider:
    """Synthetic walk/wave clip used when no GPU worker is configured."""

    id = "stub"

    async def generate(
        self,
        prompt: str,
        duration_s: float,
        seed: int,
        cfg_scale: float = 5.0,
    ) -> MotionClip:
        del cfg_scale
        rng = np.random.default_rng(seed)
        duration_s = min(max(duration_s, 0.5), 12.0)
        n_frames = max(int(duration_s * DEFAULT_FPS), 8)
        n_joints = len(SMPLH_JOINT_NAMES)
        quats = identity_quats(n_frames, n_joints)
        t = np.linspace(0.0, duration_s, n_frames, dtype=np.float64)
        phase = 2.0 * math.pi * t / 1.1

        # We encode a walk in the legs and a light opposite arm swing.
        _swing(quats, SMPLH_JOINT_INDEX["L_Hip"], phase, axis=2, amp=0.45)
        _swing(quats, SMPLH_JOINT_INDEX["R_Hip"], phase + math.pi, axis=2, amp=0.45)
        _swing(quats, SMPLH_JOINT_INDEX["L_Knee"], np.maximum(0.0, np.sin(phase)), axis=2, amp=0.7)
        _swing(quats, SMPLH_JOINT_INDEX["R_Knee"], np.maximum(0.0, np.sin(phase + math.pi)), axis=2, amp=0.7)
        _swing(quats, SMPLH_JOINT_INDEX["L_Shoulder"], phase + math.pi, axis=2, amp=0.35)
        _swing(quats, SMPLH_JOINT_INDEX["R_Shoulder"], phase, axis=2, amp=0.35)
        _swing(quats, SMPLH_JOINT_INDEX["L_Elbow"], 0.6 + 0.2 * np.sin(phase), axis=2, amp=0.5)
        _swing(quats, SMPLH_JOINT_INDEX["R_Elbow"], 0.6 + 0.2 * np.sin(phase + math.pi), axis=2, amp=0.5)
        _swing(quats, SMPLH_JOINT_INDEX["Spine1"], np.sin(phase * 0.5), axis=1, amp=0.08)
        _swing(quats, SMPLH_JOINT_INDEX["Neck"], np.sin(phase * 0.5), axis=1, amp=0.06)

        lowered = prompt.lower()
        if "wave" in lowered:
            _swing(quats, SMPLH_JOINT_INDEX["R_Shoulder"], t * 0 + 1.2, axis=2, amp=1.1)
            _swing(quats, SMPLH_JOINT_INDEX["R_Elbow"], np.sin(2.0 * math.pi * t * 2.0), axis=0, amp=0.8)

        jitter = 0.01 * rng.normal(size=(n_frames, 3))
        trans = np.stack(
            [
                0.35 * t + jitter[:, 0],
                0.92 + 0.03 * np.sin(phase * 2.0) + jitter[:, 1],
                jitter[:, 2],
            ],
            axis=1,
        )
        return MotionClip.from_numpy(
            joint_names=list(SMPLH_JOINT_NAMES),
            root_trans=trans,
            rotations_quat=quats,
            fps=DEFAULT_FPS,
            prompt=prompt,
            seed=seed,
            model_id="stub",
        )

    async def health(self) -> ProviderStatus:
        return ProviderStatus(ok=True, model_id="stub", variant="synthetic", detail="stub")


def _swing(quats: np.ndarray, joint: int, amount: np.ndarray, *, axis: int, amp: float) -> None:
    """Write a single-axis rotation into `quats` for one joint."""
    angle = amp * np.asarray(amount, dtype=np.float64)
    half = angle * 0.5
    quats[:, joint, 0] = np.cos(half)
    quats[:, joint, 1:] = 0.0
    quats[:, joint, axis + 1] = np.sin(half)
