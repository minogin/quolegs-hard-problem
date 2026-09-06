"""Where does the S/O content difference live? Per-output-dimension and per-probe-input analysis
of the paired difference S - O on the fixed probe set, over all cached baseline runs."""
import glob, pickle, sys
import numpy as np
from scipy import stats
from quolegs.introspection import make_probe_inputs
from quolegs.models import mlp_forward_numpy

runs = [pickle.load(open(p, "rb")) for p in sorted(glob.glob("results/runs_baseline/*.pkl"))]
probe = make_probe_inputs()
DS, DO = [], []
for r in runs:
    for i in (0, 1):
        for name in r.model_names[i]:
            out = mlp_forward_numpy(r.params[i][name], probe)[0]
            (DS if r.wirings[i][name]["input"] == "internal_state" else DO).append(out)
S, O = np.stack(DS), np.stack(DO)          # (400, 256, 5)
D = S - O
print("pairs", len(S))
names = ["cos x", "sin x", "cos y", "sin y", "energy"]
print("\nmean(S-O) per output dim, with paired t-test over 400 pairs (averaged over probe inputs):")
for k in range(5):
    d = D[:, :, k].mean(1)
    print(f"  {names[k]:7s} mean={d.mean():+.5f}  sd={d.std():.5f}  t={d.mean()/(d.std()/np.sqrt(len(d))):+.2f}")
print("\nmean |S-O| per output dim:", np.abs(D).mean((0, 1)).round(4))
print("mean |S-truth-ish| scale: sqrt(err S) ~ 0.028")

# which probe inputs discriminate most (energy output)
dE = D[:, :, 4]                           # (400, 256)
t = dE.mean(0) / (dE.std(0) / np.sqrt(len(dE)))
food_at = probe[:, 10:].reshape(-1, 5, 5)
energy_in = probe[:, 4]
act = probe[:, 5:10].argmax(1)
# food in the cell the action leads to
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
food_target = np.array([food_at[i, 2 + MOVES[act[i]][0], 2 + MOVES[act[i]][1]] for i in range(len(probe))])
print("\nenergy-output difference S-O by probe features (mean over pairs and probes in group):")
for lab, m in [("food at target cell", food_target > 0), ("no food at target", food_target == 0),
               ("action=stay", act == 0), ("action=move", act > 0),
               ("energy_in < 0.3", energy_in < 0.3), ("energy_in > 0.7", energy_in > 0.7),
               ("any food in window", food_at.reshape(len(probe), -1).sum(1) > 0),
               ("no food in window", food_at.reshape(len(probe), -1).sum(1) == 0)]:
    print(f"  {lab:22s} n={m.sum():3d}  mean(S-O)={dE[:, m].mean():+.5f}  |t| max={np.abs(t[m]).max():.1f}")
print("\ncorr(t-stat per probe, energy_in) =", np.corrcoef(t, energy_in)[0, 1].round(3))
print("corr(t-stat per probe, food_target) =", np.corrcoef(t, food_target)[0, 1].round(3))
# Predicted energy level: is O more optimistic?
print("\nmean predicted energy on probe: S=%.4f  O=%.4f  true next energy (no food: e-0.02, food: e+0.3 capped) = %.4f"
      % (S[:, :, 4].mean(), O[:, :, 4].mean(),
         np.minimum(1.0, energy_in - 0.02 + 0.3 * food_target).mean()))
truth_E = np.minimum(1.0, energy_in - 0.02 + 0.3 * food_target)
print("RMS error vs truth on probe (energy): S=%.4f  O=%.4f" % (np.sqrt(((S[:, :, 4] - truth_E) ** 2).mean()), np.sqrt(((O[:, :, 4] - truth_E) ** 2).mean())))
print("RMS error vs truth (energy), no-food probes: S=%.4f O=%.4f ; food probes: S=%.4f O=%.4f" % (
    np.sqrt(((S[:, food_target == 0, 4] - truth_E[food_target == 0]) ** 2).mean()), np.sqrt(((O[:, food_target == 0, 4] - truth_E[food_target == 0]) ** 2).mean()),
    np.sqrt(((S[:, food_target > 0, 4] - truth_E[food_target > 0]) ** 2).mean()), np.sqrt(((O[:, food_target > 0, 4] - truth_E[food_target > 0]) ** 2).mean())))
