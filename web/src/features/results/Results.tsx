import { num, pct } from "../../shared/format";
import type { Split } from "../../shared/data/report";
import t from "../../shared/ui/table.module.css";

const NAMES: Record<string, string> = {
  uniform: "Built for identical GPUs",
  roofline: "Roofline",
  roofline_calibrated: "Calibrated roofline",
  platform: "The platform",
  oracle: "Oracle",
};

const ROWS: [string, string, (x: number | null) => string][] = [
  ["Requests served within the SLO", "attainment", (x) => pct(x)],
  ["Requests that missed the SLO", "violated_share", (x) => pct(x, 2)],
  ["Requests shed by admission control", "shed_share", (x) => pct(x)],
  ["Replicas out of memory (replica-hours)", "crashes", (x) => num(x)],
  ["GPU-hours used by inference", "gpu_utilisation", (x) => pct(x, 0)],
  ["Requests within SLO per GPU-hour", "met_per_gpu_hour", (x) => num(x)],
  ["12 h after a driver update: within SLO", "drift_window_attainment", (x) => pct(x)],
  ["GPU-hours busy, inference + fine-tuning", "fleet_busy", (x) => pct(x, 0)],
  ["Fine-tuning tokens processed, billions", "training_tokens_b", (x) => num(x, 1)],
  ["Fine-tuning jobs preempted", "training_preemptions", (x) => num(x)],
];

export function Results({ split }: { split: Split }) {
  return (
    <div className={t.wrap}>
      <table className={t.table}>
        <thead>
          <tr>
            <th scope="col">Per site-week, {split.seeds.length} sites</th>
            {split.main.map((n) => (
              <th key={n} scope="col" className={n === "platform" ? t.us : undefined}>
                {NAMES[n] ?? n}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ROWS.map(([label, key, fmt]) => (
            <tr key={key}>
              <th scope="row">{label}</th>
              {split.main.map((n) => (
                <td key={n} className={n === "platform" ? t.us : undefined}>
                  {fmt(split.designs[n]?.[key] ?? null)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
