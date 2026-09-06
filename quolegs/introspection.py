"""Introspection boundary.

Everything below this line sees models ONLY as objects: parameter tensors, outputs on fixed probe inputs,
and error / update histories. `ModelView` is the boundary type: it has no name, no wiring, no stream.
Nothing in this module imports `wiring`.

Two families of features are computed here:
  - content  (static):  what the model *is*   -> E2 expects self/other to be indistinguishable
  - dynamics (temporal): how the model *behaves* -> E3 expects self/other to be distinguishable
Design notes: docs/decisions.md (D8)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .env import N_ACTIONS, STATE_DIM, onehot, pos_enc
from .models import mlp_forward_numpy

PROBE_SEED = 12345
PAD = 64


def make_probe_inputs(n_probe=256, n=16, k=5, food_p=0.08, seed=PROBE_SEED) -> np.ndarray:
    """Fixed probe set in the agent-model input format (state, action one-hot, food window).

    Same seed everywhere -> every model in every run is asked the same 256 questions, so their
    answers can be compared directly."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, n, size=(n_probe, 2))
    energy = rng.random(n_probe).astype(np.float32)
    st = np.stack([np.concatenate([pos_enc(p, n), [e]]) for p, e in zip(pos, energy)]).astype(np.float32)
    act = onehot(rng.integers(0, N_ACTIONS, size=n_probe))
    food = (rng.random((n_probe, k * k)) < food_p).astype(np.float32)
    return np.concatenate([st, act, food], axis=1).astype(np.float32)


@dataclass
class ModelView:
    """A model as an object. No label, no role, no wiring. This is all introspection ever gets."""
    params: list                       # [W1,b1,W2,b2,W3,b3]
    probe_out: Optional[np.ndarray]    # (n_probe, out_dim) or None if the probe does not fit the input
    h1: Optional[np.ndarray]           # hidden activations on probe (n_probe, hidden)
    h2: Optional[np.ndarray]
    err: np.ndarray                    # prediction error history (T,)
    upd: np.ndarray                    # parameter update magnitude history (T,)


def make_view(params, probe_x, err, upd) -> ModelView:
    """The only constructor. Takes raw arrays, runs the numpy forward pass itself."""
    W1 = params[0]
    if probe_x is not None and probe_x.shape[1] == W1.shape[1]:
        out, h1, h2 = mlp_forward_numpy(params, probe_x)
    else:
        out = h1 = h2 = None
    return ModelView([np.asarray(p) for p in params], out, h1, h2,
                     np.asarray(err, dtype=np.float64), np.asarray(upd, dtype=np.float64))


# ---------------------------------------------------------------------------
# Content features (static).
#
# Two nets can compute the same function with their hidden neurons renumbered; their weight matrices
# then look different although nothing meaningful differs. Every feature here is therefore invariant to
# neuron permutation: singular values, sorted vectors, and the function's outputs on the probe set.
# ---------------------------------------------------------------------------
def _pad_sorted(v, L=PAD, desc=True):
    v = np.sort(np.asarray(v, dtype=np.float64).ravel())
    if desc:
        v = v[::-1]
    out = np.zeros(L)
    m = min(L, len(v))
    out[:m] = v[:m]
    return out


def content_features(view: ModelView, with_probe=True) -> np.ndarray:
    """Permutation-invariant summary of what the model is.

    with_probe=False drops the probe outputs; needed for the W-vs-S control, where the probe set does
    not fit W's input and only the shape-agnostic part (spectra, biases, activation stats) is comparable."""
    W1, b1, W2, b2, W3, b3 = view.params
    feats = []
    for W in (W1, W2, W3):
        s = np.linalg.svd(W, compute_uv=False)                 # spectrum: order-free "size" of the matrix
        feats.append(_pad_sorted(s, 32))
        feats.append([np.linalg.norm(W), np.abs(W).mean(), W.std(), np.abs(W).max()])
    for b in (b1, b2, b3):
        feats.append(_pad_sorted(b, PAD))                       # biases, sorted
        feats.append([b.mean(), b.std()])
    if view.h1 is not None:
        for h in (view.h1, view.h2):
            feats.append(_pad_sorted(h.mean(0), PAD))           # per-neuron mean activation, sorted
            feats.append(_pad_sorted(h.std(0), PAD))
            feats.append([(h.max(0) <= 0).mean(), h.mean(), h.std()])   # dead-neuron fraction etc.
    if with_probe and view.probe_out is not None:
        feats.append(view.probe_out.ravel())                    # the function itself, on 256 fixed inputs
        feats.append(view.probe_out.mean(0))
        feats.append(view.probe_out.std(0))
    return np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in feats])


# ---------------------------------------------------------------------------
# Dynamic features: summaries of the error / update time series. Optionally
# aligned with the agent's own command stream (the efference copy).
# ---------------------------------------------------------------------------
def _acf(x, lags):
    """Autocorrelation: how similar the series is to itself shifted by `lag` steps."""
    x = x - x.mean()
    d = np.dot(x, x) + 1e-12
    return [np.dot(x[:-l], x[l:]) / d if l < len(x) else 0.0 for l in lags]


def _xcorr(x, y, lags):
    """corr(x_t, y_{t+lag}) for each lag (negative lag: y leads x)."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    out = []
    for l in lags:
        if l >= 0:
            a, b = x[:len(x) - l], y[l:]
        else:
            a, b = x[-l:], y[:len(y) + l]
        out.append(float(np.mean(a * b)) if len(a) > 2 else 0.0)
    return out


def _slope(x):
    t = np.arange(len(x), dtype=np.float64)
    t -= t.mean()
    return float(np.dot(t, x - x.mean()) / (np.dot(t, t) + 1e-12))


def dynamic_features(view: ModelView, own_actions: Optional[np.ndarray] = None, start_frac=0.5) -> np.ndarray:
    """Summary of how the model behaves over the second half of the run (models already trained).

    Without `own_actions`: intrinsic dynamics only (statistics, autocorrelations, spikes, trends of the
    error and update series). With `own_actions` (the agent's own commands, which it has anyway):
    how the error and the updates line up with *my* commands. By the theory this is where the
    asymmetry lives: one model's error is coupled to my command, the other's is not (D8)."""
    T = len(view.err)
    s = int(T * start_frac)
    err = view.err[s:]
    upd = view.upd[s:]
    lerr = np.log(err + 1e-8)
    feats = [
        lerr.mean(), lerr.std(), *np.quantile(lerr, [0.1, 0.5, 0.9, 0.99]),
        upd.mean(), upd.std(), *np.quantile(upd, [0.1, 0.5, 0.9, 0.99]),
        *_acf(lerr, [1, 2, 3, 5, 10, 20]), *_acf(upd, [1, 2, 5, 10]),
        *_xcorr(lerr, upd, [-2, -1, 0, 1, 2]),
        float((err > 10 * np.median(err)).mean()), _slope(lerr), _slope(upd),
    ]
    if own_actions is not None:
        a = np.asarray(own_actions)[s:]
        ach = (a[1:] != a[:-1]).astype(np.float64)              # 1 where my command changed
        ach = np.concatenate([[0.0], ach])
        # Does the error react to my command changing, and with what delay? ("delay between action
        # and change of error" in the spec.)
        feats += _xcorr(lerr, ach, [-3, -2, -1, 0, 1, 2, 3])
        # Does my command react to the model updating? (A model that drives the planner changes my
        # next move when it updates.)
        feats += _xcorr(upd, ach, [-3, -2, -1, 0, 1, 2, 3])
        # Error conditional on my command: for a model wired to me the error depends on what I ordered;
        # for a model wired to the other it does not.
        m = lerr.mean()
        feats += [float(lerr[a == k].mean() - m) if np.any(a == k) else 0.0 for k in range(N_ACTIONS)]
        feats += [float(upd[a == k].mean() - upd.mean()) if np.any(a == k) else 0.0 for k in range(N_ACTIONS)]
    return np.asarray(feats, dtype=np.float64)


def dynamic_feature_names(with_actions: bool):
    names = ["lerr_mean", "lerr_std", "lerr_q10", "lerr_q50", "lerr_q90", "lerr_q99",
             "upd_mean", "upd_std", "upd_q10", "upd_q50", "upd_q90", "upd_q99"]
    names += [f"acf_lerr_{l}" for l in [1, 2, 3, 5, 10, 20]] + [f"acf_upd_{l}" for l in [1, 2, 5, 10]]
    names += [f"xc_lerr_upd_{l}" for l in [-2, -1, 0, 1, 2]] + ["spike_rate", "slope_lerr", "slope_upd"]
    if with_actions:
        names += [f"xc_lerr_ach_{l}" for l in [-3, -2, -1, 0, 1, 2, 3]]
        names += [f"xc_upd_ach_{l}" for l in [-3, -2, -1, 0, 1, 2, 3]]
        names += [f"lerr_given_a{k}" for k in range(N_ACTIONS)] + [f"upd_given_a{k}" for k in range(N_ACTIONS)]
    return names


def function_distance(view_a: ModelView, view_b: ModelView) -> float:
    """RMS distance between the two models' outputs on the shared probe set (the E1b metric)."""
    return float(np.sqrt(np.mean((view_a.probe_out - view_b.probe_out) ** 2)))
