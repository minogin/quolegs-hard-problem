"""Run every experiment and write results/report.md. One command:

    python run_all.py            # full run (M=200 runs for E2/E3)
    python run_all.py --quick    # small smoke run
    python run_all.py --stages e1,e23
"""
import argparse
import json
import os
import time

import numpy as np

from quolegs.config import RunConfig, with_updates
from quolegs.wiring import DEFAULT_WIRING


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", default="swap,e1,e23,e4,e5")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", type=int, default=None, help="number of E2/E3 runs (default 200, quick 24)")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    quick = args.quick
    M = args.runs or (24 if quick else 200)
    steps = args.steps or (2000 if quick else 4000)
    n_perm = 20 if quick else 100
    e1_seeds = range(4) if quick else range(10)
    out = args.out
    os.makedirs(out, exist_ok=True)
    base = RunConfig(seed=0, steps=steps)
    stages = args.stages.split(",")
    results = {"config": base.to_dict(), "quick": quick, "M": M, "steps": steps}
    summary_path = os.path.join(out, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            results.update({k: v for k, v in json.load(f).items() if k not in results})

    def save():
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=1, default=_json_default)

    t0 = time.time()
    baseline_runs = None
    if "swap" in stages:
        from quolegs.experiments.swap import run_swap_test
        res, _ = run_swap_test(os.path.join(out, "swap"), base, range(8 if quick else 20), args.workers)
        results["swap"] = res
        print("SWAP:", json.dumps(res, indent=1, default=_json_default), flush=True)
        save()

    if "e1" in stages:
        from quolegs.experiments.e1 import run_e1
        res = run_e1(os.path.join(out, "e1"), base, e1_seeds, train_steps=steps,
                     e1a_steps=1500, e1b_steps=2000 if quick else 3000, workers=args.workers)
        results["e1"] = res
        print("E1:", json.dumps({k: v for k, v in res.items() if k != "plots"}, indent=1, default=_json_default), flush=True)
        save()

    if "e23" in stages or "e4" in stages or "e5" in stages:
        from quolegs.experiments.common import run_batch
        jobs = [(with_updates(base, seed=1000 + s), [DEFAULT_WIRING, DEFAULT_WIRING], {}) for s in range(M)]
        baseline_runs = run_batch(jobs, args.workers, cache_dir=os.path.join(out, "runs_baseline"), label="baseline")

    if "e23" in stages:
        from quolegs.experiments.e23 import run_e23
        res = run_e23(baseline_runs, os.path.join(out, "e23"), n_perm=n_perm)
        results["e23"] = res
        print("E2/E3:", json.dumps({k: v for k, v in res.items() if k != "plots"}, indent=1, default=_json_default), flush=True)
        save()

    if "e4" in stages:
        from quolegs.experiments.e4 import run_e4
        res = run_e4(baseline_runs, os.path.join(out, "e4"), base, quick=quick, workers=args.workers)
        results["e4"] = res
        print("E4:", json.dumps({k: v for k, v in res.items() if k != "plots"}, indent=1, default=_json_default), flush=True)
        save()

    if "e5" in stages:
        from quolegs.experiments.e5 import run_e5
        res = run_e5(os.path.join(out, "e5"), base, n_runs=12 if quick else 60, workers=args.workers)
        results["e5"] = res
        print("E5:", json.dumps({k: v for k, v in res.items() if k != "plots"}, indent=1, default=_json_default), flush=True)
        save()

    from quolegs.report import write_report
    path = write_report(results, out)
    print(f"report: {path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
