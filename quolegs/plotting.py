import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def smooth(x, w=50):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def plot_series(series: dict, path, title="", ylabel="", vlines=(), logy=False, w=50):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name, y in series.items():
        ax.plot(np.arange(len(smooth(y, w))) + w // 2, smooth(y, w), label=name, lw=1.2)
    for x in vlines:
        ax.axvline(x, color="k", ls="--", lw=0.8)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_curves(curves: dict, path, title="", xlabel="step", ylabel="", logy=False):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, (x, ys) in curves.items():
        ys = np.asarray(ys)
        if ys.ndim == 2:
            m, lo, hi = ys.mean(0), ys.min(0), ys.max(0)
            ax.plot(x, m, label=name, lw=1.5)
            ax.fill_between(x, lo, hi, alpha=0.15)
        else:
            ax.plot(x, ys, label=name, lw=1.5)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_accuracy_bars(items: dict, path, title="", chance=0.5):
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(items)), 3.8))
    names = list(items)
    accs = [items[n]["acc"] for n in names]
    err = np.array([[items[n]["acc"] - items[n]["ci"][0], items[n]["ci"][1] - items[n]["acc"]] for n in names]).T
    ax.bar(range(len(names)), accs, yerr=err, capsize=3, color="#4c72b0")
    ax.axhline(chance, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("held-out accuracy")
    ax.set_title(title)
    return _save(fig, path)


def plot_pca(X, y, path, title="", labels=("other", "self")):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    P = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for c, lab in enumerate(labels):
        m = np.asarray(y) == c
        ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.6, label=lab)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    return _save(fig, path)


def plot_hist_pair(a, b, path, title="", labels=("A", "B"), bins=30, xlabel=""):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    lo, hi = min(np.min(a), np.min(b)), max(np.max(a), np.max(b))
    ax.hist(a, bins=bins, range=(lo, hi), alpha=0.5, label=labels[0])
    ax.hist(b, bins=bins, range=(lo, hi), alpha=0.5, label=labels[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    return _save(fig, path)


def plot_importance(names, values, path, title="", top=15):
    idx = np.argsort(-np.abs(values))[:top]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(idx)), np.asarray(values)[idx][::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1], fontsize=8)
    ax.set_title(title)
    return _save(fig, path)
