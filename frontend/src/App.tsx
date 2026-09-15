import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { createJob, fetchHealth, fetchJob, mapSkeleton } from "./api";
import type { Health, MappingResult, MotionClip, RetargetedClip, SkeletonSnapshot } from "./types";
import { clipFromRetarget, downloadBuffer, exportAnimatedGlb } from "./viewer/clip";
import { createMannequin } from "./viewer/mannequin";
import { captureBindPose, extractSnapshot, resetToBindPose } from "./viewer/skeleton";
import { buildPlaybackClip } from "./viewer/retarget";
import { StudioCanvas } from "./viewer/StudioCanvas";

export default function App() {
  const characterRef = useRef<THREE.Object3D>(createMannequin());
  const [character, setCharacter] = useState<THREE.Object3D>(characterRef.current);
  const [snapshot, setSnapshot] = useState<SkeletonSnapshot>(() => extractSnapshot(characterRef.current));
  const [mapping, setMapping] = useState<MappingResult | null>(null);
  const [override, setOverride] = useState<Record<string, string>>({});
  const [health, setHealth] = useState<Health | null>(null);
  const [prompt, setPrompt] = useState("A person walks forward, then waves with the right hand.");
  const [duration, setDuration] = useState(4);
  const [seed, setSeed] = useState(42);
  const [cfg, setCfg] = useState(5);
  const [zeroXz, setZeroXz] = useState(true);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const actionRef = useRef<THREE.AnimationAction | null>(null);
  const clipRef = useRef<THREE.AnimationClip | null>(null);
  const [mixer, setMixer] = useState<THREE.AnimationMixer | null>(null);
  const [hasClip, setHasClip] = useState(false);
  const [clipDuration, setClipDuration] = useState(0);
  const [armSpread, setArmSpread] = useState(0);
  const [fileLabel, setFileLabel] = useState("Default mannequin");
  const [draggingFile, setDraggingFile] = useState(false);
  const sourceClipRef = useRef<MotionClip | null>(null);
  const fallbackClipRef = useRef<RetargetedClip | null>(null);
  const mappingRef = useRef<MappingResult | null>(null);

  const remap = useCallback(
    async (snap: SkeletonSnapshot, ov: Record<string, string>) => {
      const result = await mapSkeleton(snap, Object.keys(ov).length ? ov : undefined);
      setMapping(result);
    },
    [],
  );

  useEffect(() => {
    captureBindPose(characterRef.current);
    void fetchHealth()
      .then(setHealth)
      .catch((err: Error) => setError(err.message));
    void remap(snapshot, override);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const applyCharacter = useCallback(
    async (obj: THREE.Object3D, label: string) => {
      mixerRef.current?.stopAllAction();
      mixerRef.current = null;
      actionRef.current = null;
      clipRef.current = null;
      sourceClipRef.current = null;
      fallbackClipRef.current = null;
      mappingRef.current = null;
      setMixer(null);
      setHasClip(false);
      setPlaying(false);
      setClipDuration(0);
      obj.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
      characterRef.current = obj;
      setCharacter(obj);
      setFileLabel(label);
      captureBindPose(obj);
      const snap = extractSnapshot(obj);
      setSnapshot(snap);
      setOverride({});
      await remap(snap, {});
    },
    [remap],
  );

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (!lower.endsWith(".glb") && !lower.endsWith(".gltf") && !lower.endsWith(".fbx")) {
      setError("Drop a .glb, .gltf, or .fbx file");
      return;
    }
    setError("");
    const url = URL.createObjectURL(file);
    try {
      let obj: THREE.Object3D;
      if (lower.endsWith(".fbx")) {
        obj = await new FBXLoader().loadAsync(url);
      } else {
        const gltf = await new GLTFLoader().loadAsync(url);
        obj = gltf.scene;
      }
      await applyCharacter(obj, file.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      URL.revokeObjectURL(url);
    }
  };

  const onDropZoneDrag = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const generate = async () => {
    setError("");
    setBusy(true);
    setStatus("Queued…");
    try {
      mixerRef.current?.stopAllAction();
      resetToBindPose(characterRef.current);
      const snap = extractSnapshot(characterRef.current);
      setSnapshot(snap);
      const job = await createJob({
        prompt,
        duration_s: duration,
        seed,
        cfg_scale: cfg,
        snapshot: snap,
        override: Object.keys(override).length ? override : undefined,
        zero_root_xz: zeroXz,
      });
      setMapping(job.mapping);
      let current = job;
      while (current.status === "pending" || current.status === "running") {
        setStatus(current.status === "pending" ? "Queued…" : "Generating motion…");
        await sleep(800);
        current = await fetchJob(job.id);
      }
      if (current.status === "error" || (!current.source && !current.clip)) {
        throw new Error(current.error || "generation failed");
      }
      playClip(current.source, current.clip, current.mapping);
      setStatus(`Done · ${current.motion_meta?.model_id ?? "clip"}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("");
    } finally {
      setBusy(false);
    }
  };

  const playClip = (
    source: MotionClip | null,
    fallback: RetargetedClip | null,
    map: MappingResult,
    spreadDeg = armSpread,
    resume?: { time: number; playing: boolean },
  ) => {
    sourceClipRef.current = source;
    fallbackClipRef.current = fallback;
    mappingRef.current = map;
    mixerRef.current?.stopAllAction();
    resetToBindPose(characterRef.current);
    const threeClip = source
      ? buildPlaybackClip(source, map, characterRef.current, zeroXz, spreadDeg)
      : fallback
        ? clipFromRetarget(fallback, characterRef.current)
        : null;
    if (!threeClip) return;
    clipRef.current = threeClip;
    const mixer = new THREE.AnimationMixer(characterRef.current);
    mixerRef.current = mixer;
    setMixer(mixer);
    const action = mixer.clipAction(threeClip);
    action.loop = THREE.LoopRepeat;
    action.play();
    const time = Math.min(resume?.time ?? 0, threeClip.duration);
    action.time = time;
    mixer.update(0);
    if (resume && !resume.playing) {
      action.paused = true;
    }
    actionRef.current = action;
    setClipDuration(threeClip.duration);
    setHasClip(true);
    setPlaying(resume ? resume.playing : true);
    setTime(time);
  };

  const onArmSpread = (deg: number) => {
    setArmSpread(deg);
    const source = sourceClipRef.current;
    const map = mappingRef.current;
    if (!source || !map) return;
    playClip(source, fallbackClipRef.current, map, deg, {
      time: actionRef.current?.time ?? 0,
      playing,
    });
  };

  const onScrub = (value: number) => {
    const action = actionRef.current;
    const mixer = mixerRef.current;
    if (!action || !mixer) return;
    action.time = value;
    mixer.update(0);
    setTime(value);
  };

  const download = async () => {
    try {
      const buffer = await exportAnimatedGlb(characterRef.current, clipRef.current);
      downloadBuffer(buffer, "movement-smith-animation.glb");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const boneOptions = useMemo(() => snapshot.bones.map((b) => b.name), [snapshot]);

  const onOverride = (source: string, target: string) => {
    const next = { ...override };
    if (target) next[source] = target;
    else delete next[source];
    setOverride(next);
    void remap(snapshot, next);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">movement-smith</div>
        <span className={`chip ${health?.worker_ok || health?.stub || health?.modal ? "ok" : "warn"}`}>
          {health
            ? health.stub
              ? "stub motion"
              : health.modal
                ? "Modal · on demand"
                : health.worker_ok
                  ? health.worker_host ?? "worker"
                  : "worker offline"
            : "connecting"}
        </span>
        <label className="chip">
          <input type="checkbox" checked={showSkeleton} onChange={(e) => setShowSkeleton(e.target.checked)} /> skeleton
        </label>
        <div className="spacer" />
        <button type="button" className="ghost" onClick={() => void applyCharacter(createMannequin(), "Default mannequin")}>
          Reset mannequin
        </button>
        <button type="button" onClick={() => void download()} disabled={!hasClip}>
          Download GLB
        </button>
      </header>

      <div className="workspace">
        <aside className="left">
          <label
            className={`drop${draggingFile ? " active" : ""}`}
            onDragEnter={(e) => {
              onDropZoneDrag(e);
              setDraggingFile(true);
            }}
            onDragOver={onDropZoneDrag}
            onDragLeave={(e) => {
              onDropZoneDrag(e);
              if (e.currentTarget.contains(e.relatedTarget as Node)) return;
              setDraggingFile(false);
            }}
            onDrop={(e) => {
              onDropZoneDrag(e);
              setDraggingFile(false);
              void onFile(e.dataTransfer.files?.[0]);
            }}
          >
            Drop a rigged .glb or .fbx
            <div style={{ marginTop: 6, color: "#ccc" }}>{fileLabel}</div>
            <input
              type="file"
              accept=".glb,.gltf,.fbx"
              onChange={(e) => {
                void onFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </label>
          <div>
            <div className="section-title">Coverage</div>
            {mapping ? (
              <div className="coverage">
                Body <strong>{pct(mapping.body_coverage)}</strong> · Hands {pct(mapping.hand_coverage)}
                {mapping.warning ? <div className="warn-text">{mapping.warning}</div> : null}
                <div style={{ color: "#777", marginTop: 4 }}>{mapping.extra_targets.length} extra bones at rest</div>
              </div>
            ) : (
              <div className="status">Mapping…</div>
            )}
          </div>
          <div>
            <div className="section-title">Bones</div>
            <ul className="bone-list">
              {snapshot.bones.map((b) => (
                <li key={b.name}>{b.name}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="section-title">Mapping</div>
            <table className="mapping">
              <thead>
                <tr>
                  <th>SMPL-H</th>
                  <th>Target</th>
                </tr>
              </thead>
              <tbody>
                {(mapping?.joint_order ?? mapping?.pairs.map((p) => p.source) ?? []).map((source) => {
                  const target = mapping?.source_to_target[source] ?? "";
                  return (
                    <tr key={source}>
                      <td>{source}</td>
                      <td>
                        <select value={target} onChange={(e) => onOverride(source, e.target.value)}>
                          <option value="">—</option>
                          {boneOptions.map((name) => (
                            <option key={name} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </aside>
        <div className="viewport">
          <StudioCanvas
            character={character}
            mixer={mixer}
            playing={playing}
            showSkeleton={showSkeleton}
          />
        </div>
      </div>

      <footer className="promptbar">
        <div>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} maxLength={2000} />
          <div className={`status ${error ? "err" : ""}`}>{error || status}</div>
        </div>
        <div className="controls">
          <label>
            Duration {duration.toFixed(1)}s
            <input
              type="range"
              min={0.5}
              max={12}
              step={0.5}
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
            />
          </label>
          <label>
            Seed
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
          <label>
            CFG {cfg.toFixed(1)}
            <input
              type="range"
              min={1}
              max={15}
              step={0.5}
              value={cfg}
              onChange={(e) => setCfg(Number(e.target.value))}
            />
          </label>
          <label>
            Arm spread {armSpread}°
            <input
              type="range"
              min={0}
              max={90}
              step={1}
              value={armSpread}
              onChange={(e) => onArmSpread(Number(e.target.value))}
            />
          </label>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
            <input type="checkbox" checked={zeroXz} onChange={(e) => setZeroXz(e.target.checked)} />
            In-place (lock root)
          </label>
          <button type="button" onClick={() => void generate()} disabled={busy}>
            {busy ? "Generating…" : "Generate"}
          </button>
          <div className="timeline">
            <button
              type="button"
              className="ghost"
              disabled={!actionRef.current}
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={clipDuration || 1}
              step={0.01}
              value={time}
              disabled={!clipDuration}
              onChange={(e) => onScrub(Number(e.target.value))}
            />
          </div>
        </div>
      </footer>
    </div>
  );
}

function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
