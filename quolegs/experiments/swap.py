"""Wiring symmetry test: swap the S/O wiring entries without touching code. The model named "O" must
now behave as the self-model (drive the planner, predict internal_state) and efficiency must not change."""
import os

import numpy as np
from scipy import stats

from ..config import with_updates
from ..wiring import DEFAULT_WIRING, swapped
from .common import run_batch, late


def run_swap_test(out_dir, base_cfg, seeds, workers=None):
    os.makedirs(out_dir, exist_ok=True)
    jobs_def = [(with_updates(base_cfg, seed=s), [DEFAULT_WIRING, DEFAULT_WIRING], {"variant": "default"}) for s in seeds]
    jobs_sw = [(with_updates(base_cfg, seed=s), [swapped(DEFAULT_WIRING), DEFAULT_WIRING], {"variant": "swapped"}) for s in seeds]
    R_def = run_batch(jobs_def, workers, cache_dir=os.path.join(out_dir, "runs_default"), label="swap/default")
    R_sw = run_batch(jobs_sw, workers, cache_dir=os.path.join(out_dir, "runs_swapped"), label="swap/swapped")
    e_def = np.array([late(r.logs["energy"][:, 0]).mean() for r in R_def])
    e_sw = np.array([late(r.logs["energy"][:, 0]).mean() for r in R_sw])
    err_def = {n: float(np.mean([late(r.logs[f"xerr/{n}/internal_state"][:, 0]).mean() for r in R_def])) for n in ("S", "O")}
    err_sw = {n: float(np.mean([late(r.logs[f"xerr/{n}/internal_state"][:, 0]).mean() for r in R_sw])) for n in ("S", "O")}
    drv_def = [r.wirings[0] for r in R_def][0]
    drv_sw = [r.wirings[0] for r in R_sw][0]
    out = {
        "energy_default": float(e_def.mean()), "energy_swapped": float(e_sw.mean()),
        "energy_ks_p": float(stats.ks_2samp(e_def, e_sw).pvalue),
        "err_on_internal_default": err_def, "err_on_internal_swapped": err_sw,
        "driver_default": [k for k, v in drv_def.items() if v["drives_planner"]][0],
        "driver_swapped": [k for k, v in drv_sw.items() if v["drives_planner"]][0],
        "n_seeds": len(seeds),
    }
    out["pass"] = bool(out["driver_swapped"] == "O" and err_sw["O"] < err_sw["S"] and err_def["S"] < err_def["O"]
                       and out["energy_ks_p"] > 0.01)
    return out, R_def
