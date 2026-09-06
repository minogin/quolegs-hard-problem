"""E1: the self-model is a fixed point of the wiring.

E1a. Pointer switch: after training, the planner pointer moves from S to O; input wiring untouched.
E1b. Rewiring: O is fed and trained on internal_state; measure functional distance O -> S over time.
"""
import copy
import os

import numpy as np

from ..config import RunConfig
from ..introspection import make_probe_inputs
from ..sim import Simulation
from ..wiring import rewire
from .common import run_parallel
from ..plotting import plot_series, plot_curves


def _dist(agent, probe_x, real_x):
    o, s = agent.models["O"], agent.models["S"]
    d_probe = float(np.sqrt(np.mean((o.predict(probe_x) - s.predict(probe_x)) ** 2)))
    d_real = float(np.sqrt(np.mean((o.predict(real_x) - s.predict(real_x)) ** 2)))
    return d_probe, d_real


def e1_single(seed, train_steps, e1a_steps, e1b_steps, eval_every, base_cfg: dict):
    rc = RunConfig(**{k: v for k, v in base_cfg.items() if k not in ("env", "agent")})
    from ..config import EnvConfig, AgentConfig, ModelConfig
    rc.env = EnvConfig(**base_cfg["env"])
    ag = dict(base_cfg["agent"])
    ag["model"] = ModelConfig(**ag["model"])
    rc.agent = AgentConfig(**ag)
    rc.seed = seed
    probe_x = make_probe_inputs()
    sim = Simulation(rc).run(train_steps)
    trained = copy.deepcopy(sim)
    A = sim.agents[0]
    real_x = A.models["S"].buffer_sample(512)[0]
    win = min(1000, train_steps // 4)

    # ---- E1a: switch the planner pointer, keep input wiring --------------
    sim_a = copy.deepcopy(trained)
    sim_a.agents[0].set_driver("O")
    sim_a.run(e1a_steps)
    L = sim_a.logs()
    T0 = train_steps
    before, after = slice(T0 - win, T0), slice(T0 + e1a_steps - win, T0 + e1a_steps)
    e1a = {
        "energy_A_before": float(L["energy"][before, 0].mean()), "energy_A_after": float(L["energy"][after, 0].mean()),
        "energy_B_before": float(L["energy"][before, 1].mean()), "energy_B_after": float(L["energy"][after, 1].mean()),
        "deaths_A_before": float(L["died"][before, 0].sum()), "deaths_A_after": float(L["died"][after, 0].sum()),
        "errS_int_before": float(L["xerr/S/internal_state"][before, 0].mean()),
        "errS_int_after": float(L["xerr/S/internal_state"][after, 0].mean()),
        "errO_int_before": float(L["xerr/O/internal_state"][before, 0].mean()),
        "errO_int_after": float(L["xerr/O/internal_state"][after, 0].mean()),
        "errO_perc_before": float(L["xerr/O/perceived_other"][before, 0].mean()),
        "errO_perc_after": float(L["xerr/O/perceived_other"][after, 0].mean()),
        # control error (D12): prediction under *my* command vs fact on the model's own stream
        "cerrS_before": float(L["cerr/S"][before, 0].mean()), "cerrS_after": float(L["cerr/S"][after, 0].mean()),
        "cerrO_before": float(L["cerr/O"][before, 0].mean()), "cerrO_after": float(L["cerr/O"][after, 0].mean()),
        "series": {k: L[k][:, 0].copy() for k in ("energy", "xerr/S/internal_state", "xerr/O/internal_state",
                                                  "cerr/S", "cerr/O")},
    }

    # ---- E1b: rewire O onto internal_state ------------------------------
    def rewire_and_track(fresh: bool):
        s2 = copy.deepcopy(trained)
        a0 = s2.agents[0]
        a0.set_wiring(rewire(a0.wiring, "O", input="internal_state", target="internal_state"))
        a0.models["O"].clear_buffer()
        if fresh:
            a0.models["O"].reset_params(seed * 100 + 99)
        xs, dp, dr = [0], [], []
        d = _dist(a0, probe_x, real_x)
        dp.append(d[0]); dr.append(d[1])
        for k in range(e1b_steps // eval_every):
            s2.run(eval_every)
            d = _dist(a0, probe_x, real_x)
            xs.append((k + 1) * eval_every); dp.append(d[0]); dr.append(d[1])
        Lb = s2.logs()
        return {"x": np.array(xs), "d_probe": np.array(dp), "d_real": np.array(dr),
                "errO_int_final": float(Lb["xerr/O/internal_state"][-win:, 0].mean()),
                "errS_int_final": float(Lb["xerr/S/internal_state"][-win:, 0].mean())}

    # baseline distance S vs O of two *different* agents' self-models (what "different but same task" looks like)
    sB, sA = trained.agents[1].models["S"], trained.agents[0].models["S"]
    d_cross = float(np.sqrt(np.mean((sB.predict(probe_x) - sA.predict(probe_x)) ** 2)))
    e1b = {"trained": rewire_and_track(False), "fresh": rewire_and_track(True), "d_SA_SB_probe": d_cross}
    return {"seed": seed, "e1a": e1a, "e1b": e1b}


def run_e1(out_dir, base_cfg: RunConfig, seeds, train_steps, e1a_steps=1500, e1b_steps=3000,
           eval_every=100, workers=None):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [dict(seed=s, train_steps=train_steps, e1a_steps=e1a_steps, e1b_steps=e1b_steps,
                 eval_every=eval_every, base_cfg=base_cfg.to_dict()) for s in seeds]
    res = run_parallel(e1_single, jobs, workers=workers, label="E1")

    # ---- aggregate -------------------------------------------------------
    keys = [k for k in res[0]["e1a"] if k != "series"]
    e1a = {k: float(np.mean([r["e1a"][k] for r in res])) for k in keys}
    e1a_sd = {k: float(np.std([r["e1a"][k] for r in res])) for k in keys}
    drop = [r["e1a"]["energy_A_before"] - r["e1a"]["energy_A_after"] for r in res]
    e1a["energy_drop_mean"] = float(np.mean(drop))
    e1a["energy_drop_min"] = float(np.min(drop))
    e1a["n_seeds_drop_positive"] = int(np.sum(np.array(drop) > 0))
    e1a["pass"] = bool(e1a["energy_A_after"] < e1a["energy_A_before"] - 0.05
                       and e1a["errS_int_after"] < 2 * e1a["errS_int_before"] + 1e-3
                       and e1a["errO_int_after"] > 0.5 * e1a["errO_int_before"])

    x = res[0]["e1b"]["trained"]["x"]
    e1b = {}
    for var in ("trained", "fresh"):
        dp = np.stack([r["e1b"][var]["d_probe"] for r in res])
        dr = np.stack([r["e1b"][var]["d_real"] for r in res])
        e1b[var] = {"d_probe_start": float(dp[:, 0].mean()), "d_probe_end": float(dp[:, -5:].mean()),
                    "d_real_start": float(dr[:, 0].mean()), "d_real_end": float(dr[:, -5:].mean()),
                    "errO_int_final": float(np.mean([r["e1b"][var]["errO_int_final"] for r in res])),
                    "errS_int_final": float(np.mean([r["e1b"][var]["errS_int_final"] for r in res])),
                    "curve_probe": dp, "curve_real": dr}
    e1b["d_SA_SB_probe"] = float(np.mean([r["e1b"]["d_SA_SB_probe"] for r in res]))
    # convergence criterion: final real-input distance below the cross-agent self-model distance and
    # below the typical prediction error scale (sqrt of S's own error)
    err_scale = float(np.sqrt(e1b["trained"]["errS_int_final"]))
    e1b["err_scale"] = err_scale
    e1b["pass"] = bool(e1b["fresh"]["d_real_end"] < max(2 * err_scale, 0.5 * e1b["fresh"]["d_real_start"]))

    # ---- plots -----------------------------------------------------------
    plots = {}
    r0 = res[0]["e1a"]["series"]
    plots["e1a_energy"] = plot_series({"energy A": r0["energy"]}, os.path.join(out_dir, "e1a_energy.png"),
                                      title=f"E1a: planner pointer S->O at t={train_steps} (seed {res[0]['seed']})",
                                      ylabel="energy", vlines=[train_steps])
    plots["e1a_err"] = plot_series({"S on internal": r0["xerr/S/internal_state"],
                                    "O on internal": r0["xerr/O/internal_state"]},
                                   os.path.join(out_dir, "e1a_err.png"), title="E1a: error on internal_state",
                                   ylabel="MSE", vlines=[train_steps], logy=True)
    plots["e1a_cerr"] = plot_series({"S under my command": r0["cerr/S"], "O under my command": r0["cerr/O"]},
                                    os.path.join(out_dir, "e1a_control_error.png"),
                                    title="E1a: control error (D12) - does my command cause what the model predicts?",
                                    ylabel="MSE", vlines=[train_steps], logy=True)
    plots["e1b"] = plot_curves({"O(trained) vs S, real inputs": (x, e1b["trained"]["curve_real"]),
                                "O(fresh) vs S, real inputs": (x, e1b["fresh"]["curve_real"]),
                                "O(fresh) vs S, probe inputs": (x, e1b["fresh"]["curve_probe"])},
                               os.path.join(out_dir, "e1b_distance.png"), title="E1b: rewired O converges to S",
                               xlabel="steps after rewiring", ylabel="RMS output distance", logy=True)
    for var in ("trained", "fresh"):
        e1b[var].pop("curve_probe"); e1b[var].pop("curve_real")
    return {"e1a": e1a, "e1a_sd": e1a_sd, "e1b": e1b, "plots": plots, "n_seeds": len(seeds)}
