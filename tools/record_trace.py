"""Record a full trajectory (food, positions, energies, actions) of one run for the visual replay."""
import json, sys
import numpy as np
from quolegs.config import RunConfig
from quolegs.sim import Simulation

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rc = RunConfig(seed=seed, steps=steps)
sim = Simulation(rc)
frames = []

def rec(s):
    e = s.env
    frames.append({"f": np.flatnonzero(e.food.ravel()).tolist(),
                   "p": e.pos.ravel().tolist(),
                   "e": [round(float(x), 3) for x in e.energy]})

for _ in range(steps):
    sim.step([rec])
L = sim.logs()
for k, r in enumerate(L["action"]):
    frames[k]["a"] = [int(x) for x in r]
    frames[k]["d"] = [int(x) for x in L["died"][k]]
    frames[k]["errS"] = [round(float(x), 5) for x in L["err/S"][k]]
    frames[k]["errO"] = [round(float(x), 5) for x in L["err/O"][k]]
out = {"n": rc.env.n, "k": rc.env.k, "seed": seed, "frames": frames}
json.dump(out, open(sys.argv[3] if len(sys.argv) > 3 else "trace.json", "w"), separators=(",", ":"))
print("frames", len(frames), "deaths", L["died"].sum(0), "late energy", L["energy"][-1000:].mean(0))
