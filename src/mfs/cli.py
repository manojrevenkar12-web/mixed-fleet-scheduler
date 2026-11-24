"""python -m mfs.cli run dev | run heldout | gate dev | report | profile-host | bench | serve"""

from __future__ import annotations

import argparse
import json
import sys

from . import config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mfs")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every design on a split")
    r.add_argument("split", choices=["dev", "heldout"])
    r.add_argument("--workers", type=int, default=3)
    g = sub.add_parser("gate", help="exit 1 if the platform fails a gate on a split")
    g.add_argument("split", choices=["dev", "heldout"])
    sub.add_parser("report", help="regenerate REPORT.md, the README block and the console data")
    sub.add_parser("profile-host", help="measure decode-like kernels on this machine with torch.profiler")
    sub.add_parser("bench", help="measure how long one hourly placement takes")
    s = sub.add_parser("serve", help="run the placement service")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--state", default="scheduler-state.json")
    args = p.parse_args(argv)

    if args.cmd == "run":
        from .lab import study

        if args.split == "heldout":
            try:
                res = study.run_heldout_once(args.workers)
            except study.HeldOutLocked as e:
                print(e, file=sys.stderr)
                return 3
        else:
            res = study.run_split("dev", args.workers)
            study.write("dev", res)
        m = res["designs"]["platform"]
        print(f"{args.split}: attainment {m['attainment']:.4f}, violated {m['violated_share']:.4f}, gates pass: {res['gates']['platform_pass']}")
        return 0
    if args.cmd == "gate":
        res = json.loads((config.REPORTS / f"{args.split}.json").read_text(encoding="utf-8"))
        for x in res["gates"]["platform"]:
            print(f"{'pass' if x['pass'] else 'FAIL'}  {x['gate']}: {x['value']} ({x['kind']} {x['limit']})")
        return 0 if res["gates"]["platform_pass"] else 1
    if args.cmd == "report":
        from .lab.report import write_all

        for path in write_all():
            print(f"wrote {path.relative_to(config.ROOT)}")
        return 0
    if args.cmd == "profile-host":
        from .profiling.host import main as profile_main

        return profile_main()
    if args.cmd == "bench":
        from .lab.bench import main as bench_main

        return bench_main()
    if args.cmd == "serve":
        from .service import serve

        serve(args.port, args.state)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
