import { useCallback, useEffect, useState } from "react";

export type Metrics = Record<string, number | null>;

export interface Gate {
  gate: string;
  value: number;
  limit: number;
  kind: "max" | "min";
  pass: boolean;
}

export interface AccuracyRow {
  points: number;
  learned_mape: number;
  roofline_mape: number;
  interval_coverage: number;
  gpu?: string;
  extra_profiles?: number;
}

export interface Expansion {
  gpu: string;
  count: number;
  true_gain: number;
  estimators: Record<string, { predicted_gain: number; error: number | null }>;
}

export interface HourPoint {
  hour: number;
  attainment: number;
  gpus: number;
  shed: number;
  train_gpus?: number;
  preempted?: number;
}

export interface Occupancy {
  inf: Record<string, Record<string, number>>;
  train: Record<string, number>;
  drifting: string[];
  preempted: number;
}

export interface Samples {
  hours: Record<string, HourPoint[]>;
  demand_rps: number[];
  peak_hour: number;
  fleet: Record<string, number>;
  models?: string[];
  drift: { gpu: string; from_hour: number }[];
  occupancy?: Occupancy[];
}

export interface Split {
  split: string;
  seeds: number[];
  designs: Record<string, Metrics>;
  labels: Record<string, string>;
  main: string[];
  ablations: string[];
  accuracy: Record<string, AccuracyRow>;
  expansion: { per_seed: Expansion[]; mean_abs_error: Record<string, number> };
  site: { gpus: number; gpu_types: number; services: number; hours: number; profiles: number };
  gates: { platform: Gate[]; platform_pass: boolean };
  samples: Samples | Record<string, never>;
}

export interface Report {
  dev: Split;
  heldout: Split | null;
  host: { machine: { cpu: string; threads: number; torch: string }; fitted: { memory_gb_s: number; compute_gflop_s: number; memory_bound_until_batch: number } } | null;
  bench: { train_s: number; plan_ms_median: number; plan_ms_p95: number; train_points: number } | null;
}

export type Load = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; report: Report };

export function isReport(x: unknown): x is Report {
  if (typeof x !== "object" || x === null || !("dev" in x)) return false;
  const dev = (x as { dev: unknown }).dev;
  return typeof dev === "object" && dev !== null && "designs" in dev;
}

export function hasOccupancy(s: Split["samples"]): s is Samples & { occupancy: Occupancy[] } {
  return "occupancy" in s && Array.isArray(s.occupancy) && s.occupancy.length > 0;
}

export function useReport(url = "data/report.json"): { load: Load; retry: () => void } {
  const [load, setLoad] = useState<Load>({ state: "loading" });
  const [attempt, setAttempt] = useState(0);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  useEffect(() => {
    let live = true;
    setLoad({ state: "loading" });
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`The report did not load (HTTP ${r.status}). Run python -m mfs.cli report to generate it.`);
        return r.json();
      })
      .then((j: unknown) => {
        if (!live) return;
        if (!isReport(j)) throw new Error("The report file is not in the expected shape. Regenerate it with python -m mfs.cli report.");
        setLoad({ state: "ready", report: j });
      })
      .catch((e: Error) => live && setLoad({ state: "error", message: e.message }));
    return () => {
      live = false;
    };
  }, [url, attempt]);
  return { load, retry };
}
