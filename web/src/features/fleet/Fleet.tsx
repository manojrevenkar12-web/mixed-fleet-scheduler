import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { hourLabel, num, pct, serviceColor } from "../../shared/format";
import type { HourPoint, Occupancy } from "../../shared/data/report";
import styles from "./fleet.module.css";

interface Props {
  fleet: Record<string, number>;
  models: string[];
  occupancy: Occupancy[];
  hours: HourPoint[];
  drift: { gpu: string; from_hour: number }[];
  initialHour: number;
}

type Cell = { kind: "svc"; model: string } | { kind: "train" } | { kind: "idle" };

function cells(n: number, inf: Record<string, number>, train: number, models: string[]): Cell[] {
  const out: Cell[] = [];
  for (const m of models) for (let i = 0; i < (inf[m] ?? 0); i++) out.push({ kind: "svc", model: m });
  for (let i = 0; i < train; i++) out.push({ kind: "train" });
  while (out.length < n) out.push({ kind: "idle" });
  return out.slice(0, n);
}

export function Fleet({ fleet, models, occupancy, hours, drift, initialHour }: Props) {
  const [hour, setHour] = useState(initialHour);
  const [playing, setPlaying] = useState(false);
  const reduce = useReducedMotion();
  const total = Object.values(fleet).reduce((a, b) => a + b, 0);
  const last = occupancy.length - 1;

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setHour((h) => (h >= last ? 0 : h + 1)), 180);
    return () => clearInterval(t);
  }, [playing, last]);

  const occ = occupancy[Math.min(hour, last)] as Occupancy;
  const point = hours[Math.min(hour, hours.length - 1)];
  const infGpus = Object.values(occ.inf).reduce((a, g) => a + Object.values(g).reduce((x, y) => x + y, 0), 0);
  const trainGpus = Object.values(occ.train).reduce((a, b) => a + b, 0);

  const strip = useMemo(
    () =>
      occupancy.map((o, h) => {
        const inf = Object.values(o.inf).reduce((a, g) => a + Object.values(g).reduce((x, y) => x + y, 0), 0);
        const tr = Object.values(o.train).reduce((a, b) => a + b, 0);
        return { h, inf, tr, shed: (hours[h]?.shed ?? 0) > 0.001, preempted: o.preempted };
      }),
    [occupancy, hours],
  );

  return (
    <div className={styles.fleet}>
      <div className={styles.readout} aria-live="polite">
        <span className={styles.when}>{hourLabel(hour)}</span>
        <span>
          {num(infGpus)} of {num(total)} GPUs serve inference, {num(trainGpus)} fine-tune, {num(total - infGpus - trainGpus)} idle
          {point ? <>; {pct(point.attainment)} of requests within the SLO{point.shed > 0.001 ? `, ${pct(point.shed)} shed` : ""}</> : null}
          {occ.preempted > 0 ? `; ${occ.preempted} job${occ.preempted > 1 ? "s" : ""} preempted` : ""}
        </span>
      </div>

      <div className={styles.floor} role="img" aria-label={`GPU occupancy at ${hourLabel(hour)}`}>
        {Object.entries(fleet).map(([gpu, n]) => {
          const row = cells(n, occ.inf[gpu] ?? {}, occ.train[gpu] ?? 0, models);
          const drifting = occ.drifting.includes(gpu);
          const changed = drift.some((d) => d.gpu === gpu && hour >= d.from_hour);
          return (
            <div key={gpu} className={styles.row}>
              <div className={styles.gpu}>
                <span className={styles.gpuName}>{gpu}</span>
                <span className={styles.gpuCount}>{n}</span>
                {drifting ? <span className={styles.flag}>planned with a margin</span> : changed ? <span className={styles.flagQuiet}>new driver</span> : null}
              </div>
              <div className={styles.cells}>
                {row.map((c, i) => (
                  <span
                    key={i}
                    className={`${styles.cell} ${c.kind === "train" ? styles.train : c.kind === "idle" ? styles.idle : ""}`}
                    style={c.kind === "svc" ? { background: serviceColor(models, c.model) } : undefined}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <ul className={styles.legend} aria-label="Legend">
        {models.map((m) => (
          <li key={m}>
            <span className={styles.swatch} style={{ background: serviceColor(models, m) }} />
            {m}
          </li>
        ))}
        <li>
          <span className={`${styles.swatch} ${styles.train}`} />
          fine-tuning
        </li>
        <li>
          <span className={`${styles.swatch} ${styles.idle}`} />
          idle
        </li>
      </ul>

      <div className={styles.week}>
        <div className={styles.controls}>
          <button type="button" className={styles.play} onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
            {playing ? "Pause" : "Play the week"}
          </button>
          <label className={styles.scrub}>
            <span className={styles.visuallyHidden}>Hour of the week</span>
            <input type="range" min={0} max={last} value={hour} onChange={(e) => setHour(Number(e.target.value))} />
          </label>
        </div>
        <div className={styles.strip} aria-hidden="true">
          {strip.map((s) => (
            <button
              key={s.h}
              type="button"
              tabIndex={-1}
              className={styles.col}
              onClick={() => setHour(s.h)}
              title={`${hourLabel(s.h)}: ${s.inf} inference, ${s.tr} fine-tuning`}
            >
              <span className={styles.colInf} style={{ height: `${(100 * s.inf) / total}%` }} />
              <span className={`${styles.colTrain} ${styles.train}`} style={{ height: `${(100 * s.tr) / total}%` }} />
              {s.shed && <span className={styles.shedTick} />}
            </button>
          ))}
          <motion.span
            className={styles.head}
            animate={{ left: `${(100 * hour) / occupancy.length}%` }}
            transition={reduce || playing ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 38 }}
          />
          {drift.map((d) => (
            <span key={`${d.gpu}-${d.from_hour}`} className={styles.driftMark} style={{ left: `${(100 * d.from_hour) / occupancy.length}%` }}>
              {d.gpu} driver
            </span>
          ))}
        </div>
        <div className={styles.days} aria-hidden="true">
          {Array.from({ length: Math.ceil(occupancy.length / 24) }, (_, d) => (
            <span key={d}>{hourLabel(d * 24).slice(0, 3)}</span>
          ))}
        </div>
        <p className={styles.note}>
          Bars: GPUs serving inference (dark) and fine-tuning (striped) each hour, out of {total}. A red tick marks an hour in which admission control shed requests.
        </p>
      </div>
    </div>
  );
}
