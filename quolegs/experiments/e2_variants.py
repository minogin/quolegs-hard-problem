"""Mechanism check for the E2 failure (R3): does the self-model's energy pessimism come from the closed
loop "S's predictions select S's data"? Variants:
  - epsilon = 1.0 : random policy, no loop at all       -> bias should vanish, E2 -> 0.5
  - epsilon = 0.5 : weaker loop                          -> bias should shrink
  - use_other_in_planning : O also shapes A's actions (D13) -> does O acquire a bias of its own?
Each variant: N runs, then the same E2/E3 pipeline as R3 plus the probe-set energy bias S - O."""
import os
from dataclasses import replace

import numpy as np

from ..config import RunConfig, AgentConfig
from ..introspection import make_probe_inputs
from ..models import mlp_forward_numpy
from ..plotting import plot_accuracy_bars, plot_curves
from ..wiring import DEFAULT_WIRING
from .common import run_batch, late
from .e23 import run_e23


def energy_bias(results, probe):
    """Paired difference of predicted energy, self-model minus other-model, on the probe set."""
    diffs = []
    for r in results:
        for i in (0, 1):
            outs = {}
            for name in r.model_names[i]:
                outs[r.wirings[i][name]["input"]] = mlp_forward_numpy(r.params[i][name], probe)[0][:, 4]
            diffs.append((outs["internal_state"] - outs["perceived_other"]).mean())
    d = np.array(diffs)
    return {"mean": float(d.mean()), "sd": float(d.std()), "t": float(d.mean() / (d.std() / np.sqrt(len(d)))), "n": len(d)}


VARIANTS = {
    "eps0.1 (baseline)": dict(epsilon=0.1),
    "eps0.5": dict(epsilon=0.5),
    "eps1.0 (no loop)": dict(epsilon=1.0),
    "eps0.1 + use O": dict(epsilon=0.1, use_other_in_planning=True),
}


def run_e2_variants(out_dir, base: RunConfig, n_runs=100, workers=None, baseline_runs=None, n_perm=20):
    os.makedirs(out_dir, exist_ok=True)
    probe = make_probe_inputs()
    out = {}
    for label, kw in VARIANTS.items():
        tag = label.split(" ")[0] + ("_useO" if kw.get("use_other_in_planning") else "")
        if label.startswith("eps0.1 (baseline)") and baseline_runs is not None:
            runs = baseline_runs[:n_runs]
        else:
            rc = replace(base, agent=replace(base.agent, **kw))
            jobs = [(replace(rc, seed=2000 + s), [DEFAULT_WIRING, DEFAULT_WIRING], {"variant": label}) for s in range(n_runs)]
            runs = run_batch(jobs, workers, cache_dir=os.path.join(out_dir, f"runs_{tag}"), label=tag)
        e = run_e23(runs, os.path.join(out_dir, tag), n_perm=n_perm)
        out[label] = {
            "n_runs": len(runs),
            "energy": float(np.mean([late(r.logs["energy"]).mean() for r in runs])),
            "deaths_per_1000": float(np.mean([late(r.logs["died"]).sum(0).mean() for r in runs])),
            "bias": energy_bias(runs, probe),
            "e2_unpaired": e["e2"]["unpaired_full"]["acc"], "e2_paired": e["e2"]["paired_full"]["acc"],
            "e2_weights_only": e["e2"]["unpaired_spectra_biases"]["acc"],
            "e3_intrinsic": e["e3"]["unpaired_intrinsic"]["acc"], "e3_with_actions": e["e3"]["unpaired_with_actions"]["acc"],
            "streams_pass": e["streams"]["pass"], "symmetry_pass": e["symmetry"]["pass"],
        }
        print(f"[variants] {label}: {out[label]}", flush=True)
    bars = {k: {"acc": v["e2_paired"], "ci": (v["e2_paired"], v["e2_paired"])} for k, v in out.items()}
    plots = {"e2_by_variant": plot_accuracy_bars(bars, os.path.join(out_dir, "e2_by_variant.png"),
                                                 title="E2 paired content accuracy by variant")}
    labels = list(out)
    plots["bias_by_variant"] = plot_curves(
        {"energy bias S - O (probe)": (np.arange(len(labels)), np.array([out[k]["bias"]["mean"] for k in labels]))},
        os.path.join(out_dir, "bias_by_variant.png"), title="Self-model energy bias by variant",
        xlabel="variant index: " + ", ".join(f"{i}={k}" for i, k in enumerate(labels)), ylabel="mean(S - O)")
    out["plots"] = plots
    return out
