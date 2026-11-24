import { motion, useReducedMotion } from "framer-motion";
import { pct, signedPct } from "../../shared/format";
import type { Split } from "../../shared/data/report";
import t from "../../shared/ui/table.module.css";
import styles from "./model.module.css";

export function Accuracy({ split }: { split: Split }) {
  const a = split.accuracy;
  const rows: [string, string][] = [
    ["Measured pairs, other batch sizes and parallelism", "measured_pairs"],
    ["Model-on-GPU pairs never measured", "unmeasured_pairs"],
    [`A GPU type never measured (${a.unseen_gpu?.gpu ?? "?"})`, "unseen_gpu"],
    ["… after 3 calibration measurements on it", "unseen_gpu_calibrated"],
  ];
  return (
    <div className={t.wrap}>
      <table className={t.table}>
        <thead>
          <tr>
            <th scope="col">Decode-step time, error against the truth</th>
            <th scope="col">Roofline</th>
            <th scope="col" className={t.us}>
              Learned
            </th>
            <th scope="col">90% interval holds</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, k]) => {
            const r = a[k];
            if (!r) return null;
            const better = r.learned_mape < r.roofline_mape;
            return (
              <tr key={k}>
                <th scope="row">{label}</th>
                <td>{pct(r.roofline_mape)}</td>
                <td className={`${t.us} ${better ? t.pass : t.fail}`}>{pct(r.learned_mape)}</td>
                <td>{pct(r.interval_coverage, 0)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const TWINS: [string, string][] = [
  ["roofline", "Roofline twin"],
  ["learned", "Learned twin"],
  ["learned_calibrated", "Learned + 3 measurements"],
];

export function Purchase({ split }: { split: Split }) {
  const reduce = useReducedMotion();
  const ex = split.expansion.per_seed;
  const e0 = ex[0];
  if (!e0) return null;
  const max = Math.max(...ex.flatMap((e) => [e.true_gain, ...Object.values(e.estimators).map((x) => x.predicted_gain)]), 0.01);
  return (
    <div className={styles.purchase}>
      <p className={styles.question}>
        Buying {e0.count} × {e0.gpu}: how much more peak traffic does the site serve at 99% within the SLO?
      </p>
      {ex.map((e, i) => (
        <figure key={i} className={styles.site}>
          <figcaption>Site {i + 1}</figcaption>
          <div className={styles.bars}>
            {[["truth", "What it really serves", e.true_gain] as const, ...TWINS.map(([k, label]) => [k, label, e.estimators[k]?.predicted_gain ?? 0] as const)].map(([k, label, v]) => (
              <div key={k} className={styles.barRow}>
                <span className={styles.barLabel}>{label}</span>
                <span className={styles.track}>
                  <motion.span
                    className={`${styles.bar} ${k === "truth" ? styles.truth : ""}`}
                    initial={reduce ? false : { width: 0 }}
                    animate={{ width: `${(100 * Math.max(v, 0)) / max}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  />
                </span>
                <span className={styles.barValue}>{signedPct(v)}</span>
              </div>
            ))}
          </div>
        </figure>
      ))}
      <p className={styles.mae}>
        Mean absolute error of the forecast:{" "}
        {TWINS.map(([k, label]) => `${label.toLowerCase()} ${pct(split.expansion.mean_abs_error[k] ?? null, 0)}`).join(", ")}.
      </p>
    </div>
  );
}
