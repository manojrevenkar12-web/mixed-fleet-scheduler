import type { ReactNode } from "react";
import styles from "./section.module.css";

export function Section({ id, title, lead, children }: { id: string; title: string; lead?: ReactNode; children: ReactNode }) {
  return (
    <section className={styles.section} aria-labelledby={id}>
      <header className={styles.head}>
        <h2 id={id}>{title}</h2>
        {lead && <p className={styles.lead}>{lead}</p>}
      </header>
      {children}
    </section>
  );
}
