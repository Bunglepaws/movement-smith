import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";
import type { Object3D } from "three";
import defaultModelUrl from "../assets/default.glb?url";

export const DEFAULT_CHARACTER_LABEL = "Default armature";

let prototype: Object3D | null = null;
let loading: Promise<Object3D> | null = null;

async function loadPrototype(): Promise<Object3D> {
  if (prototype) return prototype;
  if (!loading) {
    loading = new GLTFLoader().loadAsync(defaultModelUrl).then((gltf) => {
      prototype = gltf.scene;
      return prototype;
    });
  }
  return loading;
}

export async function loadDefaultCharacter(): Promise<Object3D> {
  const proto = await loadPrototype();
  return cloneSkinned(proto);
}
