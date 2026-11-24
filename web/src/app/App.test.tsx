import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import fixture from "../test/report.fixture.json";
import { App } from "./App";

function serve(body: unknown, ok = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve({ ok, status: ok ? 200 : 500, json: () => Promise.resolve(body) })),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("console", () => {
  it("shows a loading state, then the fleet at the busiest hour", async () => {
    serve(fixture);
    render(<App />);
    expect(screen.getByRole("status", { name: "Loading the report" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "One site, one week" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /GPU occupancy at Mon 12:00/ })).toBeInTheDocument();
    expect(screen.getByText(/GPUs serve inference/)).toBeInTheDocument();
  });

  it("moves through the week", async () => {
    serve(fixture);
    render(<App />);
    const slider = await screen.findByRole("slider", { name: "Hour of the week" });
    fireEvent.change(slider, { target: { value: "13" } });
    expect(screen.getByRole("img", { name: /GPU occupancy at Mon 13:00/ })).toBeInTheDocument();
  });

  it("compares the designs and names the platform's column", async () => {
    serve(fixture);
    render(<App />);
    const results = await screen.findByRole("region", { name: "Against the baselines" });
    expect(within(results).getByRole("columnheader", { name: "The platform" })).toBeInTheDocument();
    expect(within(results).getByRole("columnheader", { name: "Built for identical GPUs" })).toBeInTheDocument();
    const gates = screen.getByRole("region", { name: "Release gates" });
    expect(within(gates).getByText(/release gates pass on the development sites/)).toBeInTheDocument();
  });

  it("keeps the held-out tab closed until the held-out sites have run", async () => {
    serve(fixture);
    render(<App />);
    expect(await screen.findByRole("tab", { name: "Held-out sites, run once" })).toBeDisabled();
    expect(screen.getByText("The held-out sites have not been run yet.")).toBeInTheDocument();
  });

  it("explains a failed load and loads again on request", async () => {
    serve({}, false);
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("did not load");
    serve(fixture);
    await userEvent.click(screen.getByRole("button", { name: "Load again" }));
    expect(await screen.findByRole("heading", { name: "One site, one week" })).toBeInTheDocument();
  });

  it("says so when the report has no sample week", async () => {
    serve({ ...fixture, dev: { ...fixture.dev, samples: {} } });
    render(<App />);
    expect(await screen.findByText("This report has no sample week")).toBeInTheDocument();
  });

  it("refuses a file that is not a report", async () => {
    serve({ nope: true });
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("not in the expected shape");
  });
});

describe("policies", () => {
  it("compares the scheduling policies and breaks down the GPU-hours", async () => {
    serve(fixture);
    render(<App />);
    const pol = await screen.findByRole("region", { name: "Scheduling policies, head to head" });
    expect(within(pol).getByRole("rowheader", { name: "Integer placement + elastic jobs" })).toBeInTheDocument();
    const cap = screen.getByRole("region", { name: "Where the GPU-hours went" });
    expect(within(cap).getByRole("img", { name: /Integer placement \+ elastic jobs: Serving/ })).toBeInTheDocument();
    await userEvent.click(within(cap).getByRole("button", { name: "Free: no job fits in memory" }));
    expect(within(cap).getByRole("button", { name: "Free: no job fits in memory" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("same day, 3 of 3 sites")).toBeInTheDocument();
  });

  it("says so when a report has no policy comparison", async () => {
    const old = { ...fixture, dev: { ...fixture.dev, placement_policies: undefined, start_policies: undefined, twin: undefined } };
    serve(old);
    render(<App />);
    expect(await screen.findByText("This report has no policy comparison")).toBeInTheDocument();
    expect(screen.getByText("No drift monitor in this report")).toBeInTheDocument();
  });
});
