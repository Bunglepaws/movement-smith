import type { Health, Job, MappingResult, SkeletonSnapshot } from "./types";

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

let healthInflight: Promise<Health> | null = null;
let healthCached: Health | null = null;
let healthCachedAt = 0;
const HEALTH_TTL_MS = 8000;

export function fetchHealth(): Promise<Health> {
  const now = Date.now();
  if (healthCached && now - healthCachedAt < HEALTH_TTL_MS) {
    return Promise.resolve(healthCached);
  }
  if (healthInflight) return healthInflight;
  healthInflight = fetch("/api/health")
    .then((r) => parse<Health>(r))
    .then((data) => {
      healthCached = data;
      healthCachedAt = Date.now();
      return data;
    })
    .finally(() => {
      healthInflight = null;
    });
  return healthInflight;
}

export function mapSkeleton(
  snapshot: SkeletonSnapshot,
  override?: Record<string, string>,
): Promise<MappingResult> {
  return fetch("/api/skeletons/map", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ snapshot, override }),
  }).then((r) => parse<MappingResult>(r));
}

export function createJob(body: {
  prompt: string;
  duration_s: number;
  seed: number;
  cfg_scale: number;
  snapshot: SkeletonSnapshot;
  override?: Record<string, string>;
  zero_root_xz: boolean;
}): Promise<Job> {
  return fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => parse<Job>(r));
}

export function fetchJob(id: string): Promise<Job> {
  return fetch(`/api/jobs/${id}`).then((r) => parse<Job>(r));
}
