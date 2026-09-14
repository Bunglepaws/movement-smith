import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import type { RetargetedClip } from "../types";

function wxyzToXyzw(q: number[]): [number, number, number, number] {
  return [q[1], q[2], q[3], q[0]];
}

function boneIndex(root: THREE.Object3D): Map<string, THREE.Bone> {
  const byName = new Map<string, THREE.Bone>();
  root.traverse((obj) => {
    const bone = obj as THREE.Bone;
    if (bone.isBone && bone.name && !byName.has(bone.name)) {
      byName.set(bone.name, bone);
    }
  });
  return byName;
}

/** Build a Three clip; bind by bone UUID so names with dots cannot break PropertyBinding. */
export function clipFromRetarget(data: RetargetedClip, root: THREE.Object3D): THREE.AnimationClip {
  const times: number[] = [];
  for (let i = 0; i < data.n_frames; i += 1) {
    times.push(i / data.fps);
  }
  const bones = boneIndex(root);
  const tracks: THREE.KeyframeTrack[] = [];
  for (const track of data.tracks) {
    const bone = bones.get(track.bone);
    if (!bone) continue;
    const rot = track.rotation.flatMap((q) => wxyzToXyzw(q));
    tracks.push(new THREE.QuaternionKeyframeTrack(`${bone.uuid}.quaternion`, times, rot));
    if (track.position) {
      tracks.push(new THREE.VectorKeyframeTrack(`${bone.uuid}.position`, times, track.position.flat()));
    }
  }
  return new THREE.AnimationClip("movement-smith", data.n_frames / data.fps, tracks);
}

export function exportAnimatedGlb(root: THREE.Object3D, clip: THREE.AnimationClip | null): Promise<ArrayBuffer> {
  const exporter = new GLTFExporter();
  return new Promise((resolve, reject) => {
    exporter.parse(
      root,
      (result) => {
        if (result instanceof ArrayBuffer) {
          resolve(result);
        } else {
          reject(new Error("GLTFExporter returned JSON"));
        }
      },
      (err) => reject(err),
      { binary: true, animations: clip ? [clip] : [] },
    );
  });
}

export function downloadBuffer(buffer: ArrayBuffer, filename: string): void {
  const blob = new Blob([new Uint8Array(buffer)], { type: "model/gltf-binary" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
