import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { fitCamera } from "./skeleton";

type Props = {
  character: THREE.Object3D | null;
  mixer: THREE.AnimationMixer | null;
  playing: boolean;
  showSkeleton: boolean;
};

function BoneOverlay({ object, visible }: { object: THREE.Object3D; visible: boolean }) {
  const helper = useMemo(() => new THREE.SkeletonHelper(object), [object]);
  helper.visible = visible;
  return <primitive object={helper} />;
}

function SceneContents({ character, mixer, playing, showSkeleton }: Props) {
  const { camera } = useThree();
  const fitted = useRef<THREE.Object3D | null>(null);

  useEffect(() => {
    if (character && fitted.current !== character && camera instanceof THREE.PerspectiveCamera) {
      fitCamera(character, camera);
      fitted.current = character;
    }
  }, [character, camera]);

  useFrame((_, delta) => {
    if (playing && mixer) {
      mixer.update(delta);
    }
  });

  return (
    <>
      <color attach="background" args={["#1c1c1c"]} />
      <hemisphereLight args={["#e8e4dc", "#2a2a2a", 0.9]} />
      <directionalLight
        position={[4, 8, 6]}
        intensity={1.4}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <Grid
        args={[20, 20]}
        cellSize={0.5}
        cellThickness={0.6}
        cellColor="#3a3a3a"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#555"
        fadeDistance={18}
        fadeStrength={1}
      />
      {character ? <primitive object={character} /> : null}
      {showSkeleton && character ? <BoneOverlay object={character} visible={showSkeleton} /> : null}
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
    </>
  );
}

export function StudioCanvas(props: Props) {
  return (
    <Canvas shadows camera={{ fov: 35, position: [2.4, 1.4, 3.2] }}>
      <SceneContents {...props} />
    </Canvas>
  );
}
