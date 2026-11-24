export const pct = (x: number | null | undefined, d = 1): string => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(d)}%`);
export const num = (x: number | null | undefined, d = 0): string =>
  x === null || x === undefined ? "–" : x.toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
export const signedPct = (x: number): string => `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(0)}%`;

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
export const hourLabel = (h: number): string => `${DAYS[Math.floor(h / 24) % 7]} ${String(h % 24).padStart(2, "0")}:00`;

/** Services keep their colour whatever is filtered: the order is the site's model list. */
export const serviceColor = (models: string[], model: string): string => `var(--svc-${(models.indexOf(model) % 5) + 1})`;
