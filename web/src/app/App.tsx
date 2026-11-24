import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { Fleet } from "../features/fleet/Fleet";
import { Gates } from "../features/gates/Gates";
import { Guardrails } from "../features/guardrails/Guardrails";
import { Accuracy, Purchase } from "../features/model/Model";
import { CapacityLoss, Policies, TwinDrift } from "../features/policies/Policies";
import { Results } from "../features/results/Results";
import { hasOccupancy, type Report, type Split, useReport } from "../shared/data/report";
import { num } from "../shared/format";
import { Section } from "../shared/ui/Section";
import { EmptyState, ErrorState, Skeleton } from "../shared/ui/States";
import styles from "./App.module.css";

export function App() {
  const { load, retry } = useReport();
  return (
    <main className={styles.page}>
      <header className={styles.masthead}>
        <h1>A mixed GPU fleet, placed hour by hour</h1>
        <p>
          Five GPU generations from two vendors, four LLM services with latency SLOs, and a queue of fine-tuning jobs. A PyTorch performance model learns how fast each model runs on each
          card; the scheduler places replicas, fills what is left with training, and every design is graded against a simulated truth it never sees.
        </p>
      </header>
      {load.state === "loading" && <Skeleton lines={6} label="Loading the report" />}
      {load.state === "error" && <ErrorState message={load.message} onRetry={retry} />}
      {load.state === "ready" && <Console report={load.report} />}
    </main>
  );
}

function Console({ report }: { report: Report }) {
  const [which, setWhich] = useState<"dev" | "heldout">("dev");
  const reduce = useReducedMotion();
  const split: Split = which === "heldout" && report.heldout ? report.heldout : report.dev;
  return (
    <>
      <div className={styles.tabs} role="tablist" aria-label="Which sites">
        <button type="button" role="tab" aria-selected={which === "dev"} onClick={() => setWhich("dev")}>
          Development sites
        </button>
        <button type="button" role="tab" aria-selected={which === "heldout"} onClick={() => setWhich("heldout")} disabled={!report.heldout}>
          Held-out sites, run once
        </button>
        {!report.heldout && <span className={styles.tabNote}>The held-out sites have not been run yet.</span>}
      </div>
      <AnimatePresence mode="wait">
        <motion.div
          key={which}
          initial={reduce ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduce ? undefined : { opacity: 0 }}
          transition={{ duration: 0.18 }}
        >
          <SplitView split={split} />
        </motion.div>
      </AnimatePresence>
      <footer className={styles.footer}>
        {report.bench && (
          <p>
            Measured on the build machine: training the ensemble on {report.bench.train_points} profiles takes {report.bench.train_s} s; one hourly placement takes {report.bench.plan_ms_median} ms
            (p95 {report.bench.plan_ms_p95} ms).
          </p>
        )}
        {report.host && (
          <p>
            torch.profiler on {report.host.machine.cpu} ({report.host.machine.threads} threads): {report.host.fitted.memory_gb_s} GB/s memory-bound, {report.host.fitted.compute_gflop_s} GFLOP/s
            compute-bound.
          </p>
        )}
        <p>Everything else on this page is simulated: no GPU was used.</p>
      </footer>
    </>
  );
}

function SplitView({ split }: { split: Split }) {
  const s = split.samples;
  return (
    <>
      <Section
        id="fleet"
        title="One site, one week"
        lead={`${num(split.site.gpus)} GPUs of ${split.site.gpu_types} types, re-placed every hour by the platform. Scrub or play the week; the driver updates arrive where the dashed lines are.`}
      >
        {hasOccupancy(s) ? (
          <Fleet fleet={s.fleet} models={s.models ?? Object.keys(split.designs)} occupancy={s.occupancy} hours={s.hours.platform ?? []} drift={s.drift} initialHour={s.peak_hour} />
        ) : (
          <EmptyState title="This report has no sample week">Run python -m mfs.cli run dev and python -m mfs.cli report to record one.</EmptyState>
        )}
      </Section>
      <Section id="results" title="Against the baselines" lead="The same planner in every column: only the estimate of how fast a model runs on a card, and the guardrails, differ.">
        <Results split={split} />
      </Section>
      <Section id="model" title="The performance model" lead="Mean absolute error of the predicted decode-step time. Red where the learned model is worse than the data sheet: that is where the fallback takes over.">
        <Accuracy split={split} />
      </Section>
      <Section id="purchase" title="What a purchase buys" lead="The same model, used as a twin of the site to answer a capacity question before the cards arrive.">
        <Purchase split={split} />
      </Section>
      <Section id="policies" title="Scheduling policies, head to head" lead="Greedy placement against an integer programme, and three ways to decide when a fine-tuning job starts, on the same traffic, jobs and hardware.">
        <Policies split={split} />
      </Section>
      <Section id="capacity" title="Where the GPU-hours went" lead="Every GPU-hour of the week, by what it did. Pick a part to compare it across policies. The free hours while jobs waited are split by cause, which is what pointed to elastic jobs.">
        <CapacityLoss split={split} />
      </Section>
      <Section id="drift" title="Is the Simulator still right?" lead="Every hour the Simulator's model predicts the step times the Scheduler then measures. A driver change shows up the same day.">
        <TwinDrift split={split} />
      </Section>
      <Section id="guardrails" title="What each guardrail is worth">
        <Guardrails split={split} />
      </Section>
      <Section id="gates" title="Release gates">
        <Gates split={split} />
      </Section>
    </>
  );
}
