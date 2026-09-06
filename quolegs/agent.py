"""A quoleg: world model W, two agent models (named by the wiring config), and a planner.

Nothing in this file distinguishes "self" from "other". The agent creates one model per wiring entry,
treats them identically, and asks the wiring which one the planner should consult and which stream
to start from. Design notes: docs/decisions.md (D4, D6, D10)."""
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
    """One data stream at one moment: whose state it is, the state vector, and the local observation
    centred on that actor. `internal_state` and `perceived_other` are both Sources with the same format.
    `actor` is used for one thing only: picking that actor's command as the model input (D6)."""
    actor: int
    state: np.ndarray   # (STATE_DIM,) = pos_enc(4) + energy(1)
    obs: np.ndarray     # (obs_dim,)   = food window + other-agent window


class Quoleg:
    def __init__(self, idx: int, cfg: AgentConfig, wiring: dict, obs_dim: int, seed: int):
        self.idx = idx
        self.cfg = cfg
        self.wiring = validate(copy.deepcopy(wiring))
        self.obs_dim = obs_dim
        self.food_dim = obs_dim // 2
        # World model: (window, action) -> next window.
        self.W = Predictor(obs_dim + N_ACTIONS, obs_dim, cfg.model, seed * 10 + 1)
        # One agent model per wiring entry, all the same shape: (state, action, food window) -> next state.
        # The only birth difference between them is the random init seed.
        in_dim = STATE_DIM + N_ACTIONS + self.food_dim
        self.models = {name: Predictor(in_dim, STATE_DIM, cfg.model, seed * 10 + 2 + j)
                       for j, name in enumerate(self.wiring)}
        self.rng = np.random.default_rng(seed)
        # All action sequences of length `horizon` (5^H of them), precomputed once.
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
        """Input vector of an agent model: state (5) + command (5) + food part of the window (25). See D4."""
        return np.concatenate([state, a_onehot, obs[..., :self.food_dim]], axis=-1).astype(np.float32)

    # --- planner ----------------------------------------------------------
    def plan_scores(self, model: Predictor, src: Source) -> np.ndarray:
        """Expected energy summed over the horizon, for every action sequence at once.

        Rollout: `model` predicts the next state, W predicts the next window so that on the following
        step the model can see where the food is. From step 2 on the model consumes its own predictions,
        so errors compound; hence a short horizon. Starts from whatever stream `src` is."""
        B, H = self.seq_onehot.shape[:2]
        state = np.repeat(src.state[None], B, 0)
        obs = np.repeat(src.obs[None], B, 0)
        score = np.zeros(B, dtype=np.float64)
        for h in range(H):
            a = self.seq_onehot[:, h]
            state = model.predict(self.agent_input(state, a, obs))
            score += self.gammas[h] * state[:, STATE_DIM - 1]   # last state component = energy
            if h < H - 1:
                obs = self.W.predict(np.concatenate([obs, a], axis=1))
        return score

    def act(self, sources: dict) -> int:
        """Pick an action: epsilon-greedy exploration (D10), otherwise the first step of the best plan.

        The planner consults the model flagged `drives_planner` and starts from the stream that model is
        wired to. Move the pointer to a model wired to `perceived_other` and the agent plans from the
        other's position (E1a). Ties (no food in sight) are broken at random."""
        if self.rng.random() < self.cfg.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        name = self.driver
        src = sources[self.wiring[name]["input"]]
        score = self.plan_scores(self.models[name], src)
        score = score + 1e-6 * self.rng.random(len(score))
        return int(self.seqs[int(np.argmax(score)), 0])

    # --- learning ---------------------------------------------------------
    def learn(self, sources: dict, actions, outcome: dict, obs_t: np.ndarray, obs_t1: np.ndarray) -> dict:
        """One online learning tick for all three models. Returns a log record.

        `sources`: streams at time t. `actions`: both agents' commands at t. `outcome[stream]`: the state
        that stream reports as the result of the tick (pre-respawn, see D5). Per model the order is
        error -> push -> train_step, so the logged error is measured before learning on the example."""
        rec = {}
        a_oh = onehot(np.asarray(actions))
        # World model: my window + my command -> my next window.
        xw = np.concatenate([obs_t, a_oh[self.idx]])
        rec["err/W"] = self.W.error(xw, obs_t1)
        self.W.push(xw, obs_t1)
        rec["upd/W"] = self.W.train_step()
        # Agent models: each fed from its wired input stream, with that actor's command, trained on the
        # wired target stream. The loop never looks at the model's name.
        for name, w in self.wiring.items():
            m = self.models[name]
            src = sources[w["input"]]
            x = self.agent_input(src.state, a_oh[src.actor], src.obs)
            y = outcome[w["target"]]
            # Diagnostics only (used by E1, never crosses into introspection): error on *every* stream,
            # so we can see e.g. how a model trained on the other's stream does on the own stream.
            for sname, s in sources.items():
                xs = self.agent_input(s.state, a_oh[s.actor], s.obs)
                rec[f"xerr/{name}/{sname}"] = m.error(xs, outcome[sname])
            # Control error (D12): the model's prediction under *my* command vs the fact on its own stream.
            # "Does my command cause what this model predicts?" For a model wired to me it does; for a
            # model wired to the other it does not. Logged only; not handed to introspection by default.
            xc = self.agent_input(src.state, a_oh[self.idx], src.obs)
            rec[f"cerr/{name}"] = m.error(xc, y)
            rec[f"err/{name}"] = rec[f"xerr/{name}/{w['input']}"] if w["input"] == w["target"] else m.error(x, y)
            m.push(x, y)
            rec[f"upd/{name}"] = m.train_step()
        return rec
