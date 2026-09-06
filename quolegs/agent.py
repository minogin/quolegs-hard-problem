"""A quoleg: world model W, two agent models (named by the wiring config), and a planner.

Nothing in this file distinguishes "self" from "other". Every agent model is treated identically;
the planner just asks the wiring which model to use and which source to start from."""
import copy
import itertools
from dataclasses import dataclass

import numpy as np

from .config import AgentConfig
from .env import N_ACTIONS, STATE_DIM, onehot
from .models import Predictor
from .wiring import validate, driver as wiring_driver, set_driver


@dataclass
class Source:
    """A data stream as the agent receives it: whose state it is (index), the state vector and the
    local observation centred on that actor. `actor` is only used to pick the matching command."""
    actor: int
    state: np.ndarray   # (STATE_DIM,)
    obs: np.ndarray     # (obs_dim,)


class Quoleg:
    def __init__(self, idx: int, cfg: AgentConfig, wiring: dict, obs_dim: int, seed: int):
        self.idx = idx
        self.cfg = cfg
        self.wiring = validate(copy.deepcopy(wiring))
        self.obs_dim = obs_dim
        self.food_dim = obs_dim // 2
        self.W = Predictor(obs_dim + N_ACTIONS, obs_dim, cfg.model, seed * 10 + 1)
        in_dim = STATE_DIM + N_ACTIONS + self.food_dim
        self.models = {name: Predictor(in_dim, STATE_DIM, cfg.model, seed * 10 + 2 + j)
                       for j, name in enumerate(self.wiring)}
        self.rng = np.random.default_rng(seed)
        seqs = np.array(list(itertools.product(range(N_ACTIONS), repeat=cfg.horizon)), dtype=np.int64)
        self.seqs = seqs
        self.seq_onehot = onehot(seqs)  # (B, H, 5)
        self.gammas = cfg.gamma ** np.arange(cfg.horizon)

    # --- wiring access ----------------------------------------------------
    @property
    def driver(self) -> str:
        return wiring_driver(self.wiring)

    def set_driver(self, name: str):
        self.wiring = set_driver(self.wiring, name)

    def set_wiring(self, wiring: dict):
        self.wiring = validate(copy.deepcopy(wiring))

    # --- helpers ----------------------------------------------------------
    def agent_input(self, state, a_onehot, obs):
        return np.concatenate([state, a_onehot, obs[..., :self.food_dim]], axis=-1).astype(np.float32)

    # --- planner ----------------------------------------------------------
    def plan_scores(self, model: Predictor, src: Source) -> np.ndarray:
        """Expected discounted energy for every action sequence, rolled out with W and `model`
        starting from the source the model is wired to."""
        B, H = self.seq_onehot.shape[:2]
        state = np.repeat(src.state[None], B, 0)
        obs = np.repeat(src.obs[None], B, 0)
        score = np.zeros(B, dtype=np.float64)
        for h in range(H):
            a = self.seq_onehot[:, h]
            state = model.predict(self.agent_input(state, a, obs))
            score += self.gammas[h] * state[:, STATE_DIM - 1]
            if h < H - 1:
                obs = self.W.predict(np.concatenate([obs, a], axis=1))
        return score

    def act(self, sources: dict) -> int:
        if self.rng.random() < self.cfg.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        name = self.driver
        src = sources[self.wiring[name]["input"]]
        score = self.plan_scores(self.models[name], src)
        score = score + 1e-6 * self.rng.random(len(score))  # random tie-break
        return int(self.seqs[int(np.argmax(score)), 0])

    # --- learning ---------------------------------------------------------
    def learn(self, sources: dict, actions, outcome: dict, obs_t: np.ndarray, obs_t1: np.ndarray) -> dict:
        """One online learning tick. `sources`/`outcome` are keyed by source name; `outcome[name]` is the
        state vector that stream reports for the tick's result. Returns a log record."""
        rec = {}
        a_oh = onehot(np.asarray(actions))
        xw = np.concatenate([obs_t, a_oh[self.idx]])
        rec["err/W"] = self.W.error(xw, obs_t1)
        self.W.push(xw, obs_t1)
        rec["upd/W"] = self.W.train_step()
        for name, w in self.wiring.items():
            m = self.models[name]
            src = sources[w["input"]]
            x = self.agent_input(src.state, a_oh[src.actor], src.obs)
            y = outcome[w["target"]]
            # diagnostics only (sim-level, never crosses into introspection): error on every stream
            for sname, s in sources.items():
                xs = self.agent_input(s.state, a_oh[s.actor], s.obs)
                rec[f"xerr/{name}/{sname}"] = m.error(xs, outcome[sname])
            rec[f"err/{name}"] = rec[f"xerr/{name}/{w['input']}"] if w["input"] == w["target"] else m.error(x, y)
            m.push(x, y)
            rec[f"upd/{name}"] = m.train_step()
        return rec
