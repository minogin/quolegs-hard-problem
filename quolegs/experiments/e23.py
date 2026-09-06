"""E2 (content indistinguishable) and E3 (dynamics distinguishable), plus the pre-checks:
agent symmetry and equality of the S / O training streams."""
import os

import numpy as np
from scipy import stats

from ..classify import cv_accuracy, paired_dataset, permutation_pvalue, summarize
from ..introspection import (content_features, dynamic_features, dynamic_feature_names, make_probe_inputs,
                             make_view)
from ..plotting import plot_accuracy_bars, plot_pca, plot_hist_pair, plot_importance, plot_series
from .common import late


# ---------------------------------------------------------------------------
# dataset assembly: labels come from the wiring, used ONLY by this harness
# ---------------------------------------------------------------------------
def build_views(results, probe_x):
    rows = []
    for r_idx, res in enumerate(results):
        for i in (0, 1):
            names = res.model_names[i]
            for name in names:
                w = res.wirings[i][name]
                view = make_view(res.params[i][name], probe_x, res.logs[f"err/{name}"][:, i], res.logs[f"upd/{name}"][:, i])
                rows.append({"run": r_idx, "agent": i, "name": name, "view": view,
                             "self": int(w["input"] == "internal_state"),
                             "own_actions": res.logs["action"][:, i]})
            rows.append({"run": r_idx, "agent": i, "name": "W",
                         "view": make_view(res.params[i]["W"], None, res.logs["err/W"][:, i], res.logs["upd/W"][:, i]),
                         "self": -1, "own_actions": res.logs["action"][:, i]})
    return rows


def _pairs(rows):
    """(self_row, other_row) per (run, agent)."""
    by = {}
    for r in rows:
        if r["self"] in (0, 1):
            by.setdefault((r["run"], r["agent"]), {})[r["self"]] = r
    return [(v[1], v[0]) for v in by.values() if 0 in v and 1 in v]


# ---------------------------------------------------------------------------
# pre-checks
# ---------------------------------------------------------------------------
def symmetry_check(results):
    eA = np.array([late(r.logs["energy"][:, 0]).mean() for r in results])
    eB = np.array([late(r.logs["energy"][:, 1]).mean() for r in results])
    sA = np.array([late(r.logs["err/S"][:, 0]).mean() for r in results])
    sB = np.array([late(r.logs["err/S"][:, 1]).mean() for r in results])
    dA = np.array([late(r.logs["died"][:, 0]).sum() for r in results])
    dB = np.array([late(r.logs["died"][:, 1]).sum() for r in results])
    out = {}
    for k, (a, b) in {"energy": (eA, eB), "errS": (sA, sB), "deaths": (dA, dB)}.items():
        out[k] = {"A_mean": float(a.mean()), "B_mean": float(b.mean()),
                  "ks_p": float(stats.ks_2samp(a, b).pvalue),
                  "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0}
    out["pass"] = bool(min(v["ks_p"] for v in out.values() if isinstance(v, dict)) > 0.01)
    return out, (eA, eB)


def stream_check(results):
    """Compare the S and O training buffers (inputs and targets) pooled over runs."""
    xs, ys, xo, yo = [], [], [], []
    for r in results:
        for i in (0, 1):
            for name in r.model_names[i]:
                b = r.buffers[i][name]
                if b is None:
                    continue
                if r.wirings[i][name]["input"] == "internal_state":
                    xs.append(b[0]); ys.append(b[1])
                else:
                    xo.append(b[0]); yo.append(b[1])
    xs, ys, xo, yo = map(np.concatenate, (xs, ys, xo, yo))
    d = {}
    # standardized mean differences per dimension
    def smd(a, b):
        return (a.mean(0) - b.mean(0)) / (np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-9)
    d["max_abs_smd_inputs"] = float(np.abs(smd(xs, xo)).max())
    d["max_abs_smd_targets"] = float(np.abs(smd(ys, yo)).max())
    d["ks_p_energy_input"] = float(stats.ks_2samp(xs[:, 4], xo[:, 4]).pvalue)
    d["ks_p_energy_target"] = float(stats.ks_2samp(ys[:, 4], yo[:, 4]).pvalue)
    d["action_freq_S"] = xs[:, 5:10].mean(0).round(4).tolist()
    d["action_freq_O"] = xo[:, 5:10].mean(0).round(4).tolist()
    d["food_density_S"] = float(xs[:, 10:].mean())
    d["food_density_O"] = float(xo[:, 10:].mean())
    d["n_S"] = int(len(xs)); d["n_O"] = int(len(xo))
    d["pass"] = bool(d["max_abs_smd_inputs"] < 0.1 and d["max_abs_smd_targets"] < 0.1)
    return d, (xs[:, 4], xo[:, 4])


# ---------------------------------------------------------------------------
# E2 / E3
# ---------------------------------------------------------------------------
def _eval_block(X, y, groups, label, n_perm=0, seed=0):
    res = cv_accuracy(X, y, groups, seed=seed)
    s = summarize(res)
    s["label"] = label
    if n_perm:
        p, accs = permutation_pvalue(X, y, groups, res["logreg"]["acc"], n_perm=n_perm, seed=seed)
        s["perm_p_logreg"] = p
        s["perm_null_mean"] = float(accs.mean())
        s["logreg_acc"] = res["logreg"]["acc"]
    return s


def run_e23(results, out_dir, n_perm=100, workers=None):
    os.makedirs(out_dir, exist_ok=True)
    probe_x = make_probe_inputs()
    rows = build_views(results, probe_x)
    pairs = _pairs(rows)
    SO = [r for r in rows if r["self"] in (0, 1)]
    groups = np.array([r["run"] for r in SO])
    y = np.array([r["self"] for r in SO])
    out = {"n_runs": len(results), "n_models": len(SO), "n_pairs": len(pairs)}

    sym, (eA, eB) = symmetry_check(results)
    strm, (enS, enO) = stream_check(results)
    out["symmetry"] = sym
    out["streams"] = strm
    plots = {}
    plots["sym_energy"] = plot_hist_pair(eA, eB, os.path.join(out_dir, "symmetry_energy.png"),
                                         title="Agent symmetry: late mean energy per run", labels=("A", "B"))
    plots["stream_energy"] = plot_hist_pair(enS, enO, os.path.join(out_dir, "stream_energy.png"),
                                            title="Training streams: energy input", labels=("S stream", "O stream"))

    # ---- E2: content ---------------------------------------------------
    Xc_full = np.stack([content_features(r["view"], with_probe=True) for r in SO])
    Xc_shape = np.stack([content_features(r["view"], with_probe=False) for r in SO])
    e2 = {}
    e2["unpaired_full"] = _eval_block(Xc_full, y, groups, "content, full (params+probe outputs)", n_perm=n_perm)
    e2["unpaired_shape"] = _eval_block(Xc_shape, y, groups, "content, shape-agnostic (params+activations)")
    Pa = np.stack([content_features(a["view"], True) for a, b in pairs])
    Pb = np.stack([content_features(b["view"], True) for a, b in pairs])
    D, yd, gd = paired_dataset(Pa, Pb, [a["run"] for a, b in pairs])
    e2["paired_full"] = _eval_block(D, yd, gd, "content, paired difference", n_perm=n_perm)
    # control: W vs S with the shape-agnostic features
    ctrl_rows = [r for r in rows if r["name"] == "W" or r["self"] == 1]
    Xw = np.stack([content_features(r["view"], with_probe=False, with_activations=False) for r in ctrl_rows])
    yw = np.array([int(r["self"] == 1) for r in ctrl_rows])
    gw = np.array([r["run"] for r in ctrl_rows])
    e2["control_W_vs_S"] = _eval_block(Xw, yw, gw, "control: W vs S, spectra+biases only")
    # same restricted feature set on S vs O, so the control and the test are comparable
    Xso = np.stack([content_features(r["view"], with_probe=False, with_activations=False) for r in SO])
    e2["unpaired_spectra_biases"] = _eval_block(Xso, y, groups, "content, spectra+biases only (same as control)")
    e2["headline_acc"] = max(e2["unpaired_full"]["acc"], e2["paired_full"]["acc"])
    e2["pass"] = bool(e2["headline_acc"] < 0.6 and e2["control_W_vs_S"]["acc"] > 0.9)
    out["e2"] = e2
    plots["e2_pca"] = plot_pca(Xc_full, y, os.path.join(out_dir, "e2_pca.png"), title="E2: content features (PCA)")

    # ---- E3: dynamics --------------------------------------------------
    Xd_int = np.stack([dynamic_features(r["view"], None) for r in SO])
    Xd_act = np.stack([dynamic_features(r["view"], r["own_actions"]) for r in SO])
    e3 = {}
    e3["unpaired_intrinsic"] = _eval_block(Xd_int, y, groups, "dynamics, intrinsic (err/upd series only)", n_perm=n_perm)
    e3["unpaired_with_actions"] = _eval_block(Xd_act, y, groups, "dynamics + own command stream", n_perm=n_perm)
    Pa = np.stack([dynamic_features(a["view"], a["own_actions"]) for a, b in pairs])
    Pb = np.stack([dynamic_features(b["view"], b["own_actions"]) for a, b in pairs])
    D, yd, gd = paired_dataset(Pa, Pb, [a["run"] for a, b in pairs])
    e3["paired_with_actions"] = _eval_block(D, yd, gd, "dynamics + own command, paired")
    Pa = np.stack([dynamic_features(a["view"], None) for a, b in pairs])
    Pb = np.stack([dynamic_features(b["view"], None) for a, b in pairs])
    D, yd, gd = paired_dataset(Pa, Pb, [a["run"] for a, b in pairs])
    e3["paired_intrinsic"] = _eval_block(D, yd, gd, "dynamics intrinsic, paired")
    e3["headline_acc"] = max(e3["unpaired_with_actions"]["acc"], e3["paired_with_actions"]["acc"])
    e3["pass"] = bool(e3["headline_acc"] > 0.9)
    # which features carry the signal
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(np.nan_to_num(Xd_act))
    lr = LogisticRegression(C=0.1, max_iter=5000).fit(Xs, y)
    names = dynamic_feature_names(True)
    coef = lr.coef_[0]
    top = np.argsort(-np.abs(coef))[:10]
    e3["top_features"] = [(names[i], float(coef[i])) for i in top]
    # univariate separation of the top features (self mean vs other mean, in SD units)
    e3["top_feature_smd"] = {names[i]: float((Xd_act[y == 1, i].mean() - Xd_act[y == 0, i].mean())
                                             / (Xd_act[:, i].std() + 1e-12)) for i in top[:5]}
    out["e3"] = e3
    plots["e3_importance"] = plot_importance(names, coef, os.path.join(out_dir, "e3_importance.png"),
                                             title="E3: logistic-regression weights (dynamics + own command)")
    plots["e3_pca"] = plot_pca(Xd_act, y, os.path.join(out_dir, "e3_pca.png"), title="E3: dynamic features (PCA)")
    r0 = results[0]
    plots["example_series"] = plot_series({"err S": r0.logs["err/S"][:, 0], "err O": r0.logs["err/O"][:, 0],
                                           "err W": r0.logs["err/W"][:, 0]},
                                          os.path.join(out_dir, "example_errors.png"),
                                          title="Prediction errors, agent A, seed 0", ylabel="MSE", logy=True)
    bars = {"E2 content (unpaired)": e2["unpaired_full"], "E2 content (paired)": e2["paired_full"],
            "E2 control W vs S": e2["control_W_vs_S"],
            "E3 dyn. intrinsic": e3["unpaired_intrinsic"], "E3 dyn. + own cmd": e3["unpaired_with_actions"],
            "E3 dyn. + own cmd (paired)": e3["paired_with_actions"]}
    plots["accuracies"] = plot_accuracy_bars(bars, os.path.join(out_dir, "e23_accuracies.png"),
                                             title="E2/E3: self vs other classifier accuracy")
    out["plots"] = plots
    return out
