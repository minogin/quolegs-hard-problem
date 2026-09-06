"""How many deaths are unavoidable? Hand-coded policies that know the dynamics exactly:
  - local oracle: sees its own 5x5 window, walks to the nearest visible food, random walk otherwise
  - global oracle: knows the whole map, walks to the nearest food anywhere
  - random: uniform actions
Both quolegs use the same policy. No learning involved."""
import sys
import numpy as np
from quolegs.config import RunConfig
from quolegs.env import GridWorld, MOVES

def torus_delta(a, b, n):
    d = (b - a + n // 2) % n - n // 2
    return d

def step_towards(d):
    if abs(d[0]) >= abs(d[1]) and d[0] != 0:
        return 1 if d[0] < 0 else 2
    if d[1] != 0:
        return 3 if d[1] < 0 else 4
    return 0

def policy(env, i, mode, rng):
    if mode == "random":
        return int(rng.integers(5))
    p = env.pos[i]
    cells = np.argwhere(env.food)
    if mode == "local":
        r = env.r
        cells = [c for c in cells if max(abs(torus_delta(p, c, env.n))) <= r]
    if len(cells) == 0:
        return int(rng.integers(5))
    ds = [torus_delta(p, c, env.n) for c in cells]
    best = min(ds, key=lambda d: abs(d[0]) + abs(d[1]))
    return step_towards(best)

def run(mode, seed, steps=4000):
    rc = RunConfig(seed=seed, steps=steps)
    rng = np.random.default_rng(seed)
    env = GridWorld(rc.env, rng)
    prng = np.random.default_rng(seed + 1)
    E, D = [], []
    for t in range(steps):
        a = [policy(env, i, mode, prng) for i in range(2)]
        _, died = env.step(a)
        E.append(env.energy.copy()); D.append(died.astype(float))
    E, D = np.array(E), np.array(D)
    return E[-1000:].mean(), D[-1000:].sum(0).mean()

for mode in ("random", "local", "global"):
    res = [run(mode, s) for s in range(5)]
    print(f"{mode:7s} energy={np.mean([r[0] for r in res]):.3f}  deaths/1000 steps per agent={np.mean([r[1] for r in res]):.1f}")
