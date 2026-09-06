"""Self/other classifiers with group-aware cross-validation. Used by the experiment harness only."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _models(seed):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
    }


def cv_accuracy(X, y, groups, n_splits=5, seed=0):
    """Group-wise CV accuracy for two classifier families. Returns dict name -> (acc, correct, n)."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    out = {}
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for name in _models(seed):
        correct = 0
        for tr, te in gkf.split(X, y, groups):
            clf = _models(seed)[name]
            clf.fit(X[tr], y[tr])
            correct += int((clf.predict(X[te]) == y[te]).sum())
        out[name] = {"acc": correct / len(y), "correct": correct, "n": len(y), "ci": wilson_ci(correct, len(y))}
    return out


def permutation_pvalue(X, y, groups, observed_acc, n_perm=100, seed=0, model="logreg"):
    """Permute labels within groups, recompute CV accuracy. p = P(perm acc >= observed)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    accs = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            yp[idx] = rng.permutation(yp[idx])
        accs.append(cv_accuracy_single(X, yp, groups, seed=int(rng.integers(1 << 30)), model=model))
    accs = np.asarray(accs)
    return float((accs >= observed_acc - 1e-12).mean()), accs


def cv_accuracy_single(X, y, groups, n_splits=5, seed=0, model="logreg"):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    y = np.asarray(y).astype(int)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    correct = 0
    for tr, te in gkf.split(X, y, groups):
        clf = _models(seed)[model]
        clf.fit(X[tr], y[tr])
        correct += int((clf.predict(X[te]) == y[te]).sum())
    return correct / len(y)


def paired_dataset(Xa, Xb, groups, seed=0):
    """Pairs (a, b) of feature vectors from the same run; the classifier sees the difference with a
    random sign flip and must say which one came first. Label 1 = (a - b), 0 = (b - a)."""
    rng = np.random.default_rng(seed)
    flip = rng.random(len(Xa)) < 0.5
    D = np.where(flip[:, None], Xb - Xa, Xa - Xb)
    y = (~flip).astype(int)
    return D, y, np.asarray(groups)


def summarize(res: dict) -> dict:
    best = max(res, key=lambda k: res[k]["acc"])
    return {"best": best, "acc": res[best]["acc"], "ci": res[best]["ci"], "all": {k: v["acc"] for k, v in res.items()}}
