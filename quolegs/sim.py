"""Simulation loop: two symmetric quolegs in one world. Produces RunResult with logs and model params."""
import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from .agent import Quoleg, Source
from .config import RunConfig
from .env import GridWorld, N_AGENTS, STATE_DIM
from .wiring import DEFAULT_WIRING


@dataclass
class RunResult:
    cfg: dict
    wirings: list
    logs: dict                      # name -> array (T, N_AGENTS)
    params: dict                    # agent idx -> {model name -> [W1,b1,W2,b2,W3,b3]}
    buffers: dict                   # agent idx -> {model name -> (x, y) sample}
    model_names: list               # per agent, model names in wiring order
    meta: dict = field(default_factory=dict)


class Simulation:
    def __init__(self, rc: RunConfig, wirings=None):
        torch.set_num_threads(1)
        self.rc = rc
        wirings = wirings or [DEFAULT_WIRING, DEFAULT_WIRING]
        self.wirings = [copy.deepcopy(w) for w in wirings]
        self.rng = np.random.default_rng(rc.seed)
        self.noise_rng = np.random.default_rng(rc.seed + 7_000_000)
        self.env = GridWorld(rc.env, self.rng)
        self.agents = [Quoleg(i, rc.agent, self.wirings[i], self.env.obs_dim, seed=rc.seed * 100 + i)
                       for i in range(N_AGENTS)]
        self.t = 0
        self.records = []
        self.cur = self._sources()

    # --- streams ----------------------------------------------------------
    def _noise(self, i):
        s = self.rc.agent.self_noise
        if s <= 0:
            return 0.0
        return (s * self.noise_rng.standard_normal(STATE_DIM)).astype(np.float32)

    def _sources(self):
        """Per agent: internal_state = own exact state (+ interoceptive noise if configured),
        perceived_other = the other's exact state. Same format, same quality."""
        state = [self.env.state_vec(i) for i in range(N_AGENTS)]
        obs = [self.env.obs(i) for i in range(N_AGENTS)]
        self.obs_now = obs
        out = []
        for i in range(N_AGENTS):
            j = 1 - i
            out.append({
                "internal_state": Source(i, state[i] + self._noise(i), obs[i]),
                "perceived_other": Source(j, state[j], obs[j]),
            })
        return out

    def _outcome(self, outcome_states):
        out = []
        for i in range(N_AGENTS):
            j = 1 - i
            out.append({
                "internal_state": (outcome_states[i] + self._noise(i)).astype(np.float32),
                "perceived_other": outcome_states[j],
            })
        return out

    # --- stepping ---------------------------------------------------------
    def step(self, callbacks=()):
        for cb in callbacks:
            cb(self)
        actions = [ag.act(self.cur[i]) for i, ag in enumerate(self.agents)]
        obs_t = self.obs_now
        outcome_states, died = self.env.step(actions)
        nxt = self._sources()
        outcome = self._outcome(outcome_states)
        rec = {"energy": self.env.energy.copy(), "action": np.array(actions), "died": died.astype(np.float32),
               "driver": np.array([list(ag.wiring).index(ag.driver) for ag in self.agents], dtype=np.float32)}
        per_agent = [ag.learn(self.cur[i], actions, outcome[i], obs_t[i], self.obs_now[i])
                     for i, ag in enumerate(self.agents)]
        for key in per_agent[0]:
            rec[key] = np.array([p[key] for p in per_agent], dtype=np.float32)
        self.records.append(rec)
        self.cur = nxt
        self.t += 1
        return rec

    def run(self, steps: int, callbacks=()):
        for _ in range(steps):
            self.step(callbacks)
        return self

    # --- output -----------------------------------------------------------
    def logs(self) -> dict:
        keys = self.records[0].keys()
        return {k: np.stack([r[k] for r in self.records]).astype(np.float32) for k in keys}

    def result(self, buffer_sample=256, meta=None) -> RunResult:
        params = {i: {name: m.params_numpy() for name, m in ag.models.items()} | {"W": ag.W.params_numpy()}
                  for i, ag in enumerate(self.agents)}
        buffers = {i: {name: m.buffer_sample(buffer_sample) for name, m in ag.models.items()}
                   for i, ag in enumerate(self.agents)}
        return RunResult(cfg=self.rc.to_dict(), wirings=[copy.deepcopy(ag.wiring) for ag in self.agents],
                         logs=self.logs(), params=params, buffers=buffers,
                         model_names=[list(ag.models) for ag in self.agents], meta=meta or {})


def run_simulation(rc: RunConfig, wirings=None, meta=None) -> RunResult:
    return Simulation(rc, wirings).run(rc.steps).result(meta=meta)
