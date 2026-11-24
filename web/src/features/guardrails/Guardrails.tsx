import { num, pct } from "../../shared/format";
import type { Split } from "../../shared/data/report";
import t from "../../shared/ui/table.module.css";

export function Guardrails({ split }: { split: Split }) {
  const d = split.designs;
  const base = d.platform;
  if (!base) return null;
  return (
    <div className={t.wrap}>
      <table className={t.table}>
        <thead>
          <tr>
            <th scope="col">One guardrail changed at a time</th>
            <th scope="col">Within SLO</th>
            <th scope="col">Missed SLO</th>
            <th scope="col">Inference GPU-hours</th>
            <th scope="col">After a driver update</th>
            <th scope="col">Fine-tuning tokens (bn)</th>
            <th scope="col">Preemptions</th>
          </tr>
        </thead>
        <tbody>
          {["platform", ...split.ablations].map((n) => {
            const m = d[n];
            if (!m) return null;
            return (
              <tr key={n} className={n === "platform" ? t.us : undefined}>
                <th scope="row">{n === "platform" ? "The platform" : split.labels[n]}</th>
                <td>{pct(m.attainment ?? null)}</td>
                <td>{pct(m.violated_share ?? null, 2)}</td>
                <td>{pct(m.gpu_utilisation ?? null, 0)}</td>
                <td>{pct(m.drift_window_attainment ?? null)}</td>
                <td>{num(m.training_tokens_b ?? null, 1)}</td>
                <td>{num(m.training_preemptions ?? null)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
