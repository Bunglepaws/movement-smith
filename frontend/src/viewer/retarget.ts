import * as THREE from "three";
import type { MappingResult, MotionClip, RetargetedClip } from "../types";
import { clipFromRetarget } from "./clip";
import { resetToBindPose } from "./skeleton";

const MAX_STEP_RAD = (15 * Math.PI) / 180;

/** Deform-root `spine` keeps rest rotation. Neck/head stay at bind (SMPL pitch nods the monitor). */
const SKIP_DEFORM_ROOT = new Set([
  "Pelvis",
  "Neck",
  "Head",
  "L_Collar",
  "R_Collar",
  "L_Ankle",
  "R_Ankle",
  "L_Foot",
  "R_Foot",
]);

/** SMPL spine flexion is a human crunch; we keep a fraction so the chest can sway. */
const SPINE_KEEP: Record<string, number> = {
  Spine1: 0.4,
  Spine2: 0.35,
  Spine3: 0.3,
};

/**
 * SMPL-H T-pose limb axes (Y-up). Deform-root A-pose binds already hang the
 * arms; applying L_Shoulder local as a world swing then folds them into the
 * torso. Arm bones swing from their rest direction to this axis after SMPL local.
 */
const SMPL_LIMB_REST_DIR: Record<string, THREE.Vector3> = {
  L_Shoulder: new THREE.Vector3(1, 0, 0),
  R_Shoulder: new THREE.Vector3(-1, 0, 0),
  L_Elbow: new THREE.Vector3(1, 0, 0),
  R_Elbow: new THREE.Vector3(-1, 0, 0),
};

const LIMB_CHILD_JOINT: Record<string, string> = {
  L_Shoulder: "L_Elbow",
  R_Shoulder: "R_Elbow",
  L_Elbow: "L_Wrist",
  R_Elbow: "R_Wrist",
};

const SMPLH_PARENTS = [
  -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31,
  32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
];

const SMPLH_HIP_HEIGHT_M = 0.92;

function indexBones(root: THREE.Object3D): Map<string, THREE.Bone> {
  const byName = new Map<string, THREE.Bone>();
  root.traverse((obj) => {
    const bone = obj as THREE.Bone;
    if (bone.isBone && bone.name && !byName.has(bone.name)) {
      byName.set(bone.name, bone);
    }
  });
  return byName;
}

function isHandJoint(name: string): boolean {
  return /_(Index|Middle|Pinky|Ring|Thumb)\d$/.test(name);
}

export function isMixamoRig(root: THREE.Object3D): boolean {
  let mixamo = false;
  root.traverse((obj) => {
    if (mixamo) return;
    if (obj.name.toLowerCase().includes("mixamorig")) mixamo = true;
  });
  return mixamo;
}

/**
 * Mixamo / hip-rooted humanoids (`hips`, `pelvis`) get full SMPL world deltas.
 * Deform-root spines (`DEF-spine`) do not. Blender `hips` matches this path.
 */
function useWorldDelta(root: THREE.Object3D, mapping: MappingResult): boolean {
  if (isMixamoRig(root)) return true;
  const pelvis = mapping.source_to_target.Pelvis ?? "";
  return /hip|pelvis/i.test(pelvis);
}

function smplLocalQuats(clip: MotionClip): THREE.Quaternion[][] {
  const nFrames = clip.root_trans.length;
  const nJoints = clip.joint_names.length;
  const frames: THREE.Quaternion[][] = [];
  for (let f = 0; f < nFrames; f += 1) {
    const frame: THREE.Quaternion[] = [];
    for (let j = 0; j < nJoints; j += 1) {
      const [w, x, y, z] = clip.rotations_quat[f][j];
      frame.push(new THREE.Quaternion(x, y, z, w).normalize());
    }
    frames.push(frame);
  }
  return frames;
}

function smplWorldQuats(locals: THREE.Quaternion[][]): THREE.Quaternion[][] {
  const nFrames = locals.length;
  const nJoints = locals[0]?.length ?? 0;
  const worlds: THREE.Quaternion[][] = [];
  for (let f = 0; f < nFrames; f += 1) {
    const frame: THREE.Quaternion[] = [];
    for (let j = 0; j < nJoints; j += 1) {
      const parent = j < SMPLH_PARENTS.length ? SMPLH_PARENTS[j] : -1;
      if (parent < 0) {
        frame.push(locals[f][j].clone());
      } else {
        frame.push(frame[parent].clone().multiply(locals[f][j]));
      }
    }
    worlds.push(frame);
  }
  return worlds;
}

function geodesic(a: THREE.Quaternion, b: THREE.Quaternion): number {
  return 2 * Math.acos(THREE.MathUtils.clamp(Math.abs(a.dot(b)), 0, 1));
}

function flipQuat(q: THREE.Quaternion): THREE.Quaternion {
  return q.set(-q.x, -q.y, -q.z, -q.w);
}

function slerpToward(from: THREE.Quaternion, to: THREE.Quaternion, maxRad: number): THREE.Quaternion {
  const q1 = to.clone();
  if (from.dot(q1) < 0) flipQuat(q1);
  const ang = geodesic(from, q1);
  if (ang <= maxRad || ang < 1e-8) return q1;
  return from.clone().slerp(q1, maxRad / ang);
}

/** Sign-continuity plus a per-frame speed cap so 180° source flips cannot kick a limb. */
function stabilizeLocals(locals: THREE.Quaternion[][]): THREE.Quaternion[][] {
  const nFrames = locals.length;
  const nJoints = locals[0]?.length ?? 0;
  const out = locals.map((frame) => frame.map((q) => q.clone()));
  for (let f = 1; f < nFrames; f += 1) {
    for (let j = 0; j < nJoints; j += 1) {
      if (out[f][j].dot(out[f - 1][j]) < 0) flipQuat(out[f][j]);
    }
  }
  if (nFrames >= 3) {
    for (let f = 1; f < nFrames - 1; f += 1) {
      for (let j = 0; j < nJoints; j += 1) {
        const prev = out[f - 1][j];
        const cur = out[f][j];
        const next = out[f + 1][j];
        if (geodesic(prev, cur) > MAX_STEP_RAD && geodesic(cur, next) > MAX_STEP_RAD && geodesic(prev, next) < MAX_STEP_RAD * 2) {
          out[f][j] = prev.clone().slerp(next, 0.5);
        }
      }
    }
  }
  for (let f = 1; f < nFrames; f += 1) {
    for (let j = 0; j < nJoints; j += 1) {
      out[f][j] = slerpToward(out[f - 1][j], out[f][j], MAX_STEP_RAD);
    }
  }
  return out;
}

function ensureQuatContinuity(values: number[]): void {
  for (let i = 4; i < values.length; i += 4) {
    const dot =
      values[i] * values[i - 4] +
      values[i + 1] * values[i - 3] +
      values[i + 2] * values[i - 2] +
      values[i + 3] * values[i - 1];
    if (dot < 0) {
      values[i] *= -1;
      values[i + 1] *= -1;
      values[i + 2] *= -1;
      values[i + 3] *= -1;
    }
  }
}

function tracksFromWxyz(rot: Record<string, number[]>, positions: Record<string, number[][] | null>): RetargetedClip["tracks"] {
  return Object.entries(rot).map(([bone, values]) => {
    ensureQuatContinuity(values);
    const rotation: number[][] = [];
    for (let i = 0; i < values.length; i += 4) {
      rotation.push([values[i], values[i + 1], values[i + 2], values[i + 3]]);
    }
    return { bone, rotation, position: positions[bone] ?? null };
  });
}

function restBoneDirection(
  bone: THREE.Bone,
  preferredChild: string | null,
  bones: Map<string, THREE.Bone>,
): THREE.Vector3 | null {
  const child = preferredChild ? bones.get(preferredChild) : undefined;
  if (child) {
    const dir = child.getWorldPosition(new THREE.Vector3()).sub(bone.getWorldPosition(new THREE.Vector3()));
    if (dir.lengthSq() > 1e-8) return dir.normalize();
  }
  for (const obj of bone.children) {
    const next = obj as THREE.Bone;
    if (!next.isBone) continue;
    const dir = next.getWorldPosition(new THREE.Vector3()).sub(bone.getWorldPosition(new THREE.Vector3()));
    if (dir.lengthSq() > 1e-8) return dir.normalize();
  }
  return new THREE.Vector3(0, 1, 0)
    .applyQuaternion(new THREE.Quaternion().setFromRotationMatrix(bone.matrixWorld))
    .normalize();
}

/** Swing the rest local so the bone points at `smplLocal * smplRestDir` in world. */
function limbLocalFromDirection(
  smplLocal: THREE.Quaternion,
  restWorld: THREE.Quaternion,
  parentRest: THREE.Quaternion,
  charRestDir: THREE.Vector3,
  smplRestDir: THREE.Vector3,
): THREE.Quaternion | null {
  const smplDirWorld = smplRestDir.clone().applyQuaternion(smplLocal);
  if (charRestDir.lengthSq() < 1e-8 || smplDirWorld.lengthSq() < 1e-8) return null;
  const swing = new THREE.Quaternion().setFromUnitVectors(charRestDir.clone().normalize(), smplDirWorld.normalize());
  const animWorld = swing.multiply(restWorld.clone());
  return parentRest.clone().invert().multiply(animWorld);
}

/**
 * Deform-root rigs cannot take SMPL pelvis. Each mapped bone gets only that
 * joint's parent-local rotation: `inv(R_parent_rest) * R_smpl_local * R_rest`.
 * Using full SMPL world (relative to pelvis) dumped the torso into every limb.
 * Arm bones use direction matching so A-pose binds do not fold into the chest.
 */
export function retargetDeformRoot(
  clip: MotionClip,
  mapping: MappingResult,
  root: THREE.Object3D,
  zeroRoot: boolean,
): RetargetedClip {
  resetToBindPose(root);
  const bones = indexBones(root);
  const restWorld = new Map<string, THREE.Quaternion>();
  const restLocal = new Map<string, THREE.Quaternion>();
  const parentName = new Map<string, string | null>();
  for (const [name, bone] of bones) {
    restWorld.set(name, new THREE.Quaternion().setFromRotationMatrix(bone.matrixWorld));
    restLocal.set(name, bone.quaternion.clone());
    const parent = bone.parent as THREE.Bone | null;
    parentName.set(name, parent?.isBone && bones.has(parent.name) ? parent.name : null);
  }

  const jointIndex = Object.fromEntries(clip.joint_names.map((n, i) => [n, i]));
  const apply: Record<string, string> = {};
  for (const [joint, target] of Object.entries(mapping.source_to_target)) {
    if (SKIP_DEFORM_ROOT.has(joint) || isHandJoint(joint) || !bones.has(target) || jointIndex[joint] === undefined) {
      continue;
    }
    apply[joint] = target;
  }

  const limbRestDir = new Map<string, THREE.Vector3>();
  for (const [joint, target] of Object.entries(apply)) {
    const smplDir = SMPL_LIMB_REST_DIR[joint];
    if (!smplDir) continue;
    const bone = bones.get(target);
    if (!bone) continue;
    const childJoint = LIMB_CHILD_JOINT[joint];
    const childBone = childJoint ? (mapping.source_to_target[childJoint] ?? null) : null;
    const dir = restBoneDirection(bone, childBone, bones);
    if (dir) limbRestDir.set(target, dir);
  }

  const smplLocal = stabilizeLocals(smplLocalQuats(clip));
  const nFrames = clip.root_trans.length;
  const rot: Record<string, number[]> = {};
  const positions: Record<string, number[][] | null> = {};
  for (const target of Object.values(apply)) {
    rot[target] = [];
    positions[target] = null;
  }

  const hipBone = mapping.source_to_target.Pelvis ?? null;
  if (hipBone && bones.has(hipBone) && restLocal.has(hipBone)) {
    const restL = restLocal.get(hipBone)!;
    rot[hipBone] = [];
    for (let f = 0; f < nFrames; f += 1) {
      rot[hipBone].push(restL.w, restL.x, restL.y, restL.z);
    }
    positions[hipBone] = !zeroRoot ? rootLocalPositions(clip, bones.get(hipBone)!) : null;
  }

  for (let f = 0; f < nFrames; f += 1) {
    for (const [joint, target] of Object.entries(apply)) {
      const restW = restWorld.get(target);
      const restL = restLocal.get(target);
      if (!restW || !restL) continue;
      const parent = parentName.get(target);
      const parentRest = parent
        ? restWorld.get(parent)
        : implicitParentWorld(restW, restL);
      if (!parentRest) continue;
      let smpl = smplLocal[f][jointIndex[joint]];
      const keep = SPINE_KEEP[joint];
      if (keep !== undefined) {
        smpl = new THREE.Quaternion().slerp(smpl, keep);
      }
      const smplRestDir = SMPL_LIMB_REST_DIR[joint];
      const charRestDir = limbRestDir.get(target);
      const limbLocal =
        smplRestDir && charRestDir
          ? limbLocalFromDirection(smpl, restW, parentRest, charRestDir, smplRestDir)
          : null;
      const local = limbLocal ?? parentRest.clone().invert().multiply(smpl).multiply(restW);
      rot[target].push(local.w, local.x, local.y, local.z);
    }
  }

  return {
    fps: clip.fps,
    n_frames: nFrames,
    quat_order: "wxyz",
    tracks: tracksFromWxyz(rot, positions),
    hip_bone: hipBone,
  };
}

function implicitParentWorld(restWorld: THREE.Quaternion, restLocal: THREE.Quaternion): THREE.Quaternion {
  return restWorld.clone().multiply(restLocal.clone().invert());
}

function rootLocalPositions(clip: MotionClip, hip: THREE.Bone): number[][] {
  const restLocal = hip.position.clone();
  const hipY = Math.abs(hip.getWorldPosition(new THREE.Vector3()).y);
  const scale = hipY > 1e-3 ? hipY / SMPLH_HIP_HEIGHT_M : 1;
  const origin = clip.root_trans[0];
  const invParent = new THREE.Quaternion();
  if (hip.parent) {
    hip.parent.getWorldQuaternion(invParent).invert();
  }
  return clip.root_trans.map((t) => {
    const delta = new THREE.Vector3(
      (t[0] - origin[0]) * scale,
      (t[1] - origin[1]) * scale,
      (t[2] - origin[2]) * scale,
    );
    delta.applyQuaternion(invParent);
    return [restLocal.x + delta.x, restLocal.y + delta.y, restLocal.z + delta.z];
  });
}

/** Same joints the Python retarget leaves at rest. Clavicles fold cartoon arms in. */
const SKIP_WORLD_DELTA = new Set([
  "L_Collar",
  "R_Collar",
  "L_Wrist",
  "R_Wrist",
  "L_Ankle",
  "R_Ankle",
  "L_Foot",
  "R_Foot",
]);

function topoBoneNames(bones: Map<string, THREE.Bone>, parentName: Map<string, string | null>): string[] {
  const seen = new Set<string>();
  const order: string[] = [];
  const visit = (name: string) => {
    if (seen.has(name)) return;
    seen.add(name);
    const parent = parentName.get(name);
    if (parent && bones.has(parent)) visit(parent);
    order.push(name);
  };
  for (const name of bones.keys()) visit(name);
  return order;
}

/**
 * Apply SMPL world rotations as deltas on the target bind pose:
 * `R_anim_world = R_smpl_world * R_rest_world`. Locals use the live parent world,
 * including a non-bone Armature recovered as `R_world * R_local^{-1}`.
 * Skipped bones keep rest local and follow the animated parent.
 */
export function retargetWorldDelta(
  clip: MotionClip,
  mapping: MappingResult,
  root: THREE.Object3D,
  zeroRoot: boolean,
): RetargetedClip {
  resetToBindPose(root);
  const bones = indexBones(root);
  const restWorld = new Map<string, THREE.Quaternion>();
  const restLocal = new Map<string, THREE.Quaternion>();
  const parentName = new Map<string, string | null>();
  for (const [name, bone] of bones) {
    restWorld.set(name, new THREE.Quaternion().setFromRotationMatrix(bone.matrixWorld));
    restLocal.set(name, bone.quaternion.clone());
    const parent = bone.parent as THREE.Bone | null;
    parentName.set(name, parent?.isBone && bones.has(parent.name) ? parent.name : null);
  }

  const jointIndex = Object.fromEntries(clip.joint_names.map((n, i) => [n, i]));
  const apply: Record<string, string> = {};
  const targetToJoint = new Map<string, string>();
  for (const [joint, target] of Object.entries(mapping.source_to_target)) {
    if (
      SKIP_WORLD_DELTA.has(joint) ||
      isHandJoint(joint) ||
      !bones.has(target) ||
      jointIndex[joint] === undefined
    ) {
      continue;
    }
    apply[joint] = target;
    targetToJoint.set(target, joint);
  }

  const smplWorld = smplWorldQuats(stabilizeLocals(smplLocalQuats(clip)));
  const nFrames = clip.root_trans.length;
  const rot: Record<string, number[]> = {};
  const positions: Record<string, number[][] | null> = {};
  for (const target of Object.values(apply)) {
    rot[target] = [];
    positions[target] = null;
  }

  const hipBone = apply.Pelvis ?? null;
  if (!zeroRoot && hipBone) {
    const hip = bones.get(hipBone);
    if (hip) positions[hipBone] = rootLocalPositions(clip, hip);
  }

  const order = topoBoneNames(bones, parentName);
  for (let f = 0; f < nFrames; f += 1) {
    const animWorld = new Map<string, THREE.Quaternion>();
    for (const name of order) {
      const restW = restWorld.get(name);
      const restL = restLocal.get(name);
      if (!restW || !restL) continue;
      const joint = targetToJoint.get(name);
      const j = joint !== undefined ? jointIndex[joint] : undefined;
      if (j !== undefined) {
        animWorld.set(name, smplWorld[f][j].clone().multiply(restW));
      } else {
        const parent = parentName.get(name);
        if (parent && animWorld.has(parent)) {
          animWorld.set(name, animWorld.get(parent)!.clone().multiply(restL));
        } else {
          animWorld.set(name, restW.clone());
        }
      }
    }
    for (const target of Object.values(apply)) {
      const anim = animWorld.get(target);
      const restL = restLocal.get(target);
      if (!anim || !restL) continue;
      const parent = parentName.get(target);
      const parentWorld = parent
        ? animWorld.get(parent)
        : implicitParentWorld(restWorld.get(target)!, restL);
      if (!parentWorld) continue;
      const local = parentWorld.clone().invert().multiply(anim);
      rot[target].push(local.w, local.x, local.y, local.z);
    }
  }

  return {
    fps: clip.fps,
    n_frames: nFrames,
    quat_order: "wxyz",
    tracks: tracksFromWxyz(rot, positions),
    hip_bone: hipBone,
  };
}

/** Mixamo-style hips use world deltas; deform-root spines stay at rest. */
export function buildPlaybackClip(
  clip: MotionClip,
  mapping: MappingResult,
  root: THREE.Object3D,
  zeroRoot: boolean,
): THREE.AnimationClip {
  const data = useWorldDelta(root, mapping)
    ? retargetWorldDelta(clip, mapping, root, zeroRoot)
    : retargetDeformRoot(clip, mapping, root, zeroRoot);
  return clipFromRetarget(data, root);
}
