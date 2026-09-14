import * as THREE from "three";
import type { SkeletonSnapshot } from "../types";

/** Restore GLTF/FBX bind pose so retarget never snapshots a posed frame. */
export function resetToBindPose(root: THREE.Object3D): void {
  let posed = false;
  root.traverse((obj) => {
    const mesh = obj as THREE.SkinnedMesh;
    if (mesh.isSkinnedMesh && mesh.skeleton) {
      mesh.skeleton.pose();
      posed = true;
    }
  });
  if (!posed) {
    const bind = root.userData.bindPose as Array<{
      bone: THREE.Bone;
      position: THREE.Vector3;
      quaternion: THREE.Quaternion;
      scale: THREE.Vector3;
    }> | undefined;
    if (bind) {
      for (const entry of bind) {
        entry.bone.position.copy(entry.position);
        entry.bone.quaternion.copy(entry.quaternion);
        entry.bone.scale.copy(entry.scale);
      }
    }
  }
  root.updateMatrixWorld(true);
}

export function captureBindPose(root: THREE.Object3D): void {
  resetToBindPose(root);
  const bind: Array<{
    bone: THREE.Bone;
    position: THREE.Vector3;
    quaternion: THREE.Quaternion;
    scale: THREE.Vector3;
  }> = [];
  root.traverse((obj) => {
    const bone = obj as THREE.Bone;
    if (!bone.isBone) return;
    bind.push({
      bone,
      position: bone.position.clone(),
      quaternion: bone.quaternion.clone(),
      scale: bone.scale.clone(),
    });
  });
  root.userData.bindPose = bind;
}

export function extractSnapshot(root: THREE.Object3D): SkeletonSnapshot {
  root.updateMatrixWorld(true);
  const seen = new Set<string>();
  const bones: SkeletonSnapshot["bones"] = [];

  const skinned: THREE.SkinnedMesh[] = [];
  root.traverse((obj) => {
    if ((obj as THREE.SkinnedMesh).isSkinnedMesh) {
      skinned.push(obj as THREE.SkinnedMesh);
    }
  });

  const collect = (bone: THREE.Bone) => {
    if (seen.has(bone.uuid)) return;
    seen.add(bone.uuid);
    const parent = bone.parent && (bone.parent as THREE.Bone).isBone ? bone.parent.name : null;
    bones.push({
      name: bone.name || `bone_${bones.length}`,
      parent,
      rest_local: bone.matrix.toArray(),
      rest_world: bone.matrixWorld.toArray(),
    });
  };

  if (skinned.length > 0) {
    for (const mesh of skinned) {
      for (const bone of mesh.skeleton.bones) {
        collect(bone);
      }
    }
  } else {
    root.traverse((obj) => {
      if ((obj as THREE.Bone).isBone) {
        collect(obj as THREE.Bone);
      }
    });
  }

  return { bones };
}

export function listBoneNames(root: THREE.Object3D): string[] {
  return extractSnapshot(root).bones.map((b) => b.name);
}

export function fitCamera(root: THREE.Object3D, camera: THREE.PerspectiveCamera): void {
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.5);
  camera.position.set(center.x + maxDim * 1.6, center.y + maxDim * 0.4, center.z + maxDim * 1.8);
  camera.near = maxDim / 100;
  camera.far = maxDim * 100;
  camera.lookAt(center);
  camera.updateProjectionMatrix();
}
