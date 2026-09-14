export type BoneSnapshot = {
  name: string;
  parent: string | null;
  rest_local: number[];
  rest_world: number[];
};

export type SkeletonSnapshot = {
  bones: BoneSnapshot[];
};

export type BonePair = {
  source: string;
  target: string;
  method: string;
};

export type MappingResult = {
  joint_order: string[];
  pairs: BonePair[];
  unmapped_sources: string[];
  extra_targets: string[];
  body_coverage: number;
  hand_coverage: number;
  coverage: number;
  warning: string | null;
  source_to_target: Record<string, string>;
};

export type BoneTrack = {
  bone: string;
  rotation: number[][];
  position: number[][] | null;
};

export type RetargetedClip = {
  fps: number;
  n_frames: number;
  quat_order: string;
  tracks: BoneTrack[];
  hip_bone: string | null;
};

export type MotionClip = {
  skeleton: string;
  fps: number;
  joint_names: string[];
  root_trans: number[][];
  rotations_quat: number[][][];
  prompt: string | null;
  seed: number | null;
  model_id: string | null;
};

export type Job = {
  id: string;
  status: "pending" | "running" | "done" | "error";
  prompt: string;
  duration_s: number;
  seed: number;
  mapping: MappingResult;
  error: string | null;
  clip: RetargetedClip | null;
  source: MotionClip | null;
  motion_meta: { model_id: string; fps: number; n_frames: number; duration_s: number } | null;
  created_at: string;
};

export type Health = {
  ok: boolean;
  provider: string;
  stub: boolean;
  worker_ok: boolean;
  worker_host: string | null;
  model_id: string;
  variant: string | null;
  detail: string | null;
  /** False when the API skips remote /v1/health (Modal on-demand). */
  probe_worker?: boolean;
  modal?: boolean;
};
