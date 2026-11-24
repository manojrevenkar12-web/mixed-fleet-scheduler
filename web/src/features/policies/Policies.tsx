import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import type { Split } from "../../shared/data/report";
import { num, pct } from "../../shared/format";
import { EmptyState } from "../../shared/ui/States";
import t from "../../shared/ui/table.module.css";
import s from "./policies.module.css";

const SHORT: Record<string, string> = {
  platform: "Greedy placement, look-ahead rule",
  platform_ilp: "Integer placement",
  naive_backfill: "Any free GPU",
  learned_backfill: "Learned start/wait policy",
  elastic_jobs: "Elastic jobs",
  ilp_elastic: "Integer placement + elastic jobs",
};

export const PARTS: { key: string[]; label: string; tone: string }[] = [
  { key: ["serving"], label: "Serving", tone: "serve" },
  { key: ["inference_headroom"], label: "Inference headroom", tone: "headroom" },
  { key: ["training_kept"], label: "Fine-tuning kept", tone: "train" },
  { key: ["training_lost_to_preemption", "training_out_of_memory"], label: "Fine-tuning lost", tone: "lost" },
  { key: ["stranded_job_fit_but_waited"], label: "Free: the policy waited", tone: "waited" },
  { key: ["stranded_too_few_gpus_free"], label: "Free: too few GPUs", tone: "few" },
  { key: ["stranded_no_job_fits_memory"], label: "Free: no job fits in memory", tone: "nofit" },
  { key: ["idle_nothing_queued"], label: "Free, nothing queued", tone: "idle" },
];

export function Policies({ split }: { split: Split }) {
  const names = [...(split.placement_policies ?? []), ...(split.start_policies ?? []).filter((n) => !(split.placement_policies ?? []).includes(n))].filter((n) => split.designs[n]);
  if (names.length === 0) return <EmptyState title="This report has no policy comparison">Run python -m mfs.cli run dev and python -m mfs.cli report with the current code.</EmptyState>;
  const d = split.designs;
  return (
    <div className={t.wrap}>
      <table className={t.table}>
        <thead>
          <tr>
            <th scope="col">Policy, same traces</th>
            <th scope="col">Within SLO</th>
            <th scope="col">Missed SLO</th>
            <th scope="col">Within SLO per GPU-hour</th>
            <th scope="col">Replica starts a day</th>
            <th scope="col">Fine-tuning tokens (bn)</th>
            <th scope="col">Jobs finished</th>
            <th scope="col">Preemptions</th>
          </tr>
        </thead>
        <tbody>
          {names.map((n) => {
            const m = d[n] ?? {};
            return (
              <tr key={n} className={n === "platform" ? t.us : undefined}>
                <th scope="row">{SHORT[n] ?? split.labels[n]}</th>
                <td>{pct(m.attainment ?? null)}</td>
                <td>{pct(m.violated_share ?? null, 2)}</td>
                <td>{num(m.met_per_gpu_hour ?? null)}</td>
                <td>{num(m.replica_starts_per_day ?? null)}</td>
                <td>{num(m.training_net_tokens_b ?? null, 1)}</td>
                <td>{pct(m.training_done_share ?? null, 0)}</td>
                <td>{num(m.training_preemptions ?? null)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function CapacityLoss({ split }: { split: Split }) {
  const names = ["platform", "naive_backfill", "learned_backfill", "elastic_jobs", "ilp_elastic"].filter((n) => split.designs[n]?.cap_serving != null);
  const [focus, setFocus] = useState<string | null>(null);
  const reduce = useReducedMotion();
  if (names.length === 0) return <EmptyState title="This report has no capacity breakdown">It is computed for every design from the hourly placements; rerun the study.</EmptyState>;
  const share = (n: string, keys: string[]) => keys.reduce((a, k) => a + (split.designs[n]?.[`cap_${k}`] ?? 0), 0);
  return (
    <div className={s.wrap}>
      <ul className={s.legend} aria-label="Where a GPU-hour went">
        {PARTS.map((p) => (
          <li key={p.label}>
            <button type="button" className={s.key} data-tone={p.tone} aria-pressed={focus === p.label} onClick={() => setFocus(focus === p.label ? null : p.label)}>
              {p.label}
            </button>
          </li>
        ))}
      </ul>
      <div className={s.bars}>
        {names.map((n, i) => (
          <div key={n} className={s.row}>
            <span className={s.name}>{SHORT[n] ?? split.labels[n]}</span>
            <div className={s.bar} role="img" aria-label={`${SHORT[n] ?? n}: ${PARTS.map((p) => `${p.label} ${pct(share(n, p.key), 0)}`).join(", ")}`}>
              {PARTS.map((p) => {
                const v = share(n, p.key);
                return v > 0 ? (
                  <motion.span
                    key={p.label}
                    className={s.seg}
                    data-tone={p.tone}
                    data-dim={focus !== null && focus !== p.label ? "" : undefined}
                    title={`${p.label}: ${pct(v, 1)}`}
                    initial={reduce ? false : { scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.35, delay: reduce ? 0 : i * 0.05 }}
                    style={{ flexGrow: v }}
                  />
                ) : null;
              })}
            </div>
            <span className={s.busy}>{pct(share(n, ["serving", "inference_headroom", "training_kept", "training_lost_to_preemption"]), 0)} busy</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function TwinDrift({ split }: { split: Split }) {
  const tw = split.twin;
  if (!tw) return <EmptyState title="No drift monitor in this report">The Simulator-vs-live comparison is recorded for the platform design; rerun the study.</EmptyState>;
  const fired = tw.alarm_days_after_change.filter((x) => x !== null);
  const sameDay = fired.filter((x) => x === 0).length;
  const when = `${sameDay === fired.length ? "same day" : "within " + Math.max(...(fired as number[])) + " d"}, ${fired.length} of ${tw.alarm_days_after_change.length} sites`;
  return (
    <dl className={s.twin}>
      <div>
        <dt>Simulator error before the driver change</dt>
        <dd>{pct(tw.twin_error_before)}</dd>
      </div>
      <div>
        <dt>... after it, on that GPU type</dt>
        <dd className={s.warn}>{pct(tw.twin_error_after)}</dd>
      </div>
      <div>
        <dt>The Scheduler's calibrated model after it</dt>
        <dd>{pct(tw.calibrated_error_after)}</dd>
      </div>
      <div>
        <dt>Alarm at {pct(tw.threshold, 0)} daily error</dt>
        <dd>{when}</dd>
      </div>
      <div>
        <dt>Alarms with no change</dt>
        <dd>{tw.false_alarm_days}</dd>
      </div>
    </dl>
  );
}
