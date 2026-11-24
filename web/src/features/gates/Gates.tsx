import type { Split } from "../../shared/data/report";
import t from "../../shared/ui/table.module.css";

export function Gates({ split }: { split: Split }) {
  const gates = split.gates.platform;
  const passed = gates.filter((g) => g.pass).length;
  return (
    <div className={t.wrap}>
      <p>
        {passed} of {gates.length} release gates pass on the {split.split === "heldout" ? "held-out" : "development"} sites.
      </p>
      <table className={t.table}>
        <thead>
          <tr>
            <th scope="col">Gate</th>
            <th scope="col">Value</th>
            <th scope="col">Limit</th>
            <th scope="col">Result</th>
          </tr>
        </thead>
        <tbody>
          {gates.map((g) => (
            <tr key={g.gate}>
              <th scope="row">{g.gate}</th>
              <td>{g.value.toFixed(3)}</td>
              <td>
                {g.kind === "min" ? "≥" : "≤"} {g.limit}
              </td>
              <td className={g.pass ? t.pass : t.fail}>{g.pass ? "pass" : "fail"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
