import * as THREE from "three";
import { FBXExporter } from "@comfyorg/fbx-exporter-three";
import { resetToBindPose } from "./skeleton";

type SavedTransform = {
  object: THREE.Object3D;
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  scale: THREE.Vector3;
};

const FBX_OPTS = {
  preset: "blender" as const,
  fps: 30,
  creator: "movement-smith",
};

function saveTransforms(root: THREE.Object3D): SavedTransform[] {
  const saved: SavedTransform[] = [];
  root.traverse((object) => {
    saved.push({
      object,
      position: object.position.clone(),
      quaternion: object.quaternion.clone(),
      scale: object.scale.clone(),
    });
  });
  return saved;
}

function restoreTransforms(saved: SavedTransform[]): void {
  for (const entry of saved) {
    entry.object.position.copy(entry.position);
    entry.object.quaternion.copy(entry.quaternion);
    entry.object.scale.copy(entry.scale);
  }
}

/** We snapshot the current pose, export bind pose, then restore playback. */
function withBindPose<T>(root: THREE.Object3D, fn: () => T): T {
  const saved = saveTransforms(root);
  try {
    resetToBindPose(root);
    return fn();
  } finally {
    restoreTransforms(saved);
    root.updateMatrixWorld(true);
  }
}

/** We write bind-pose mesh, materials, and armature with no animation stack. */
export function exportModelFbx(root: THREE.Object3D): Promise<Uint8Array> {
  const exporter = new FBXExporter();
  return withBindPose(root, () =>
    exporter.parseAsync(root, {
      ...FBX_OPTS,
      includeAnimations: false,
    }),
  );
}

/** We write armature and clip only so the file can be retargeted onto another skin. */
export function exportAnimationFbx(root: THREE.Object3D, clip: THREE.AnimationClip, fps = 30): Uint8Array {
  const exporter = new FBXExporter();
  return withBindPose(root, () =>
    exporter.parseSync(root, {
      ...FBX_OPTS,
      fps,
      includeAnimations: true,
      animations: [clip],
      objectFilter: (object) => !object.isMesh && !object.isLight && !object.isCamera,
    }),
  );
}

export function exportStem(label: string): string {
  const withoutExt = label.trim().replace(/\.(glb|gltf|fbx)$/i, "");
  const stem = withoutExt.replace(/[^\w.-]+/g, "-").replace(/^-+|-+$/g, "");
  return stem || "character";
}
