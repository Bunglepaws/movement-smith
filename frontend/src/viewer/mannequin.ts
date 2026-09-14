import * as THREE from "three";

const SKIN = 0xb8a07a;
const JOINT = 0x2b2b2b;

function addLimb(
  parent: THREE.Object3D,
  name: string,
  offset: THREE.Vector3,
  length: number,
  direction: THREE.Vector3,
  thickness: number,
  color = SKIN,
): THREE.Bone {
  const bone = new THREE.Bone();
  bone.name = name;
  bone.position.copy(offset);
  const dir = direction.clone().normalize();
  bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  parent.add(bone);

  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(thickness, length, thickness),
    new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.05 }),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.position.set(0, length / 2, 0);
  bone.add(mesh);

  const joint = new THREE.Mesh(
    new THREE.SphereGeometry(thickness * 0.45, 12, 12),
    new THREE.MeshStandardMaterial({ color: JOINT, roughness: 0.35 }),
  );
  joint.castShadow = true;
  bone.add(joint);
  return bone;
}

/** Mixamo-named puppet so the default character maps without an upload. */
export function createMannequin(): THREE.Group {
  const root = new THREE.Group();
  root.name = "mannequin";
  const up = new THREE.Vector3(0, 1, 0);
  const down = new THREE.Vector3(0, -1, 0);
  const left = new THREE.Vector3(1, 0, 0);
  const right = new THREE.Vector3(-1, 0, 0);

  const hips = addLimb(root, "Hips", new THREE.Vector3(0, 0.95, 0), 0.12, up, 0.26);
  const spine = addLimb(hips, "Spine", new THREE.Vector3(0, 0.12, 0), 0.13, up, 0.22);
  const spine1 = addLimb(spine, "Spine1", new THREE.Vector3(0, 0.13, 0), 0.13, up, 0.24);
  const spine2 = addLimb(spine1, "Spine2", new THREE.Vector3(0, 0.13, 0), 0.16, up, 0.28);
  const neck = addLimb(spine2, "Neck", new THREE.Vector3(0, 0.16, 0), 0.08, up, 0.08);
  addLimb(neck, "Head", new THREE.Vector3(0, 0.08, 0), 0.18, up, 0.16, 0xc4aa88);

  const lShoulder = addLimb(spine2, "LeftShoulder", new THREE.Vector3(0.14, 0.12, 0), 0.08, left, 0.08);
  const lArm = addLimb(lShoulder, "LeftArm", new THREE.Vector3(0, 0.08, 0), 0.26, left, 0.08);
  const lFore = addLimb(lArm, "LeftForeArm", new THREE.Vector3(0, 0.26, 0), 0.24, left, 0.07);
  addLimb(lFore, "LeftHand", new THREE.Vector3(0, 0.24, 0), 0.1, left, 0.06);

  const rShoulder = addLimb(spine2, "RightShoulder", new THREE.Vector3(-0.14, 0.12, 0), 0.08, right, 0.08);
  const rArm = addLimb(rShoulder, "RightArm", new THREE.Vector3(0, 0.08, 0), 0.26, right, 0.08);
  const rFore = addLimb(rArm, "RightForeArm", new THREE.Vector3(0, 0.26, 0), 0.24, right, 0.07);
  addLimb(rFore, "RightHand", new THREE.Vector3(0, 0.24, 0), 0.1, right, 0.06);

  const lUp = addLimb(hips, "LeftUpLeg", new THREE.Vector3(0.1, 0, 0), 0.42, down, 0.1);
  const lLeg = addLimb(lUp, "LeftLeg", new THREE.Vector3(0, 0.42, 0), 0.4, down, 0.09);
  addLimb(lLeg, "LeftFoot", new THREE.Vector3(0, 0.4, 0.06), 0.16, new THREE.Vector3(0, 0, 1), 0.08);

  const rUp = addLimb(hips, "RightUpLeg", new THREE.Vector3(-0.1, 0, 0), 0.42, down, 0.1);
  const rLeg = addLimb(rUp, "RightLeg", new THREE.Vector3(0, 0.42, 0), 0.4, down, 0.09);
  addLimb(rLeg, "RightFoot", new THREE.Vector3(0, 0.4, 0.06), 0.16, new THREE.Vector3(0, 0, 1), 0.08);

  root.updateMatrixWorld(true);
  return root;
}
