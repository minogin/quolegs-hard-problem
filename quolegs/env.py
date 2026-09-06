"""Torus grid world with food and two symmetric agents.

This module knows nothing about models, wiring or "self"/"other". It only moves agents, grows food,
drains energy and reports what happened. Design notes: docs/decisions.md (D2, D3, D5, D7)."""
import numpy as np
from .config import EnvConfig

N_ACTIONS = 5
# Action index -> (drow, dcol). Index 0 is "stay", the rest are the four directions.
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)  # stay, up, down, left, right
N_AGENTS = 2
STATE_DIM = 5  # pos_enc(4) + energy(1)


def pos_enc(pos, n: int) -> np.ndarray:
    """Torus-aware position encoding: (cos x, sin x, cos y, sin y).

    The grid wraps around, so cell 15 is next to cell 0. A raw index would make them look far apart
    to a network; mapping each coordinate to an angle on a circle keeps neighbours close (D2)."""
    a = 2 * np.pi * np.asarray(pos, dtype=np.float64) / n
    return np.array([np.cos(a[0]), np.sin(a[0]), np.cos(a[1]), np.sin(a[1])], dtype=np.float32)


def onehot(a, n=N_ACTIONS):
    """Action index -> vector of length 5 with a single 1. Standard way to feed a category to a net."""
    a = np.asarray(a)
    out = np.zeros(a.shape + (n,), dtype=np.float32)
    np.put_along_axis(out, a[..., None], 1.0, axis=-1)
    return out


class GridWorld:
    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        # All randomness comes from `rng`, which is seeded by the caller -> runs are reproducible.
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.n
        self.k = cfg.k
        self.r = cfg.k // 2
        self.obs_dim = 2 * cfg.k * cfg.k
        self.food = np.zeros((self.n, self.n), dtype=bool)          # where food lies
        self.pos = rng.integers(0, self.n, size=(N_AGENTS, 2))       # (row, col) per agent
        self.energy = np.full(N_AGENTS, cfg.energy_init, dtype=np.float64)
        self.last_action = np.zeros(N_AGENTS, dtype=np.int64)
        for _ in range(cfg.max_food // 2):
            self._spawn_one()

    # --- food -------------------------------------------------------------
    def _spawn_one(self):
        """Drop one food item on a random empty cell, unless the cap is reached."""
        if self.food.sum() >= self.cfg.max_food:
            return
        for _ in range(64):
            x, y = self.rng.integers(0, self.n, size=2)
            if not self.food[x, y]:
                self.food[x, y] = True
                return

    # --- observation ------------------------------------------------------
    def _idx(self, pos):
        """Row/col indices of the k x k window centred on `pos`; `%` wraps around the torus."""
        rows = (pos[0] - self.r + np.arange(self.k)) % self.n
        cols = (pos[1] - self.r + np.arange(self.k)) % self.n
        return np.ix_(rows, cols)

    def food_window(self, pos) -> np.ndarray:
        return self.food[self._idx(pos)].astype(np.float32)

    def obs(self, i: int) -> np.ndarray:
        """What agent i sees: two k x k layers, flattened into 2*k*k numbers.

        Layer 1: food in the window. Layer 2: a single 1 where the other agent stands, if it is inside
        the window. There is no "walls" layer because a torus has no walls (D3)."""
        p = self.pos[i]
        food = self.food_window(p)
        other = np.zeros((self.k, self.k), dtype=np.float32)
        q = self.pos[1 - i]
        d = (q - p + self.r) % self.n
        if d[0] < self.k and d[1] < self.k:
            other[d[0], d[1]] = 1.0
        return np.concatenate([food.ravel(), other.ravel()])

    def state_vec(self, i: int) -> np.ndarray:
        """The 5-number state of agent i: encoded position + energy."""
        return np.concatenate([pos_enc(self.pos[i], self.n), [np.float32(self.energy[i])]]).astype(np.float32)

    # --- dynamics ---------------------------------------------------------
    def step(self, actions):
        """Advance one tick. Returns (outcome_states, died).

        Order: move -> drain energy -> eat -> check death -> record outcome -> respawn -> grow food.

        `outcome_states[i]` is what the tick did to agent i *before* respawn, with energy clipped to 0
        on death. This is the prediction target for the agent models: they see death as "energy 0",
        never as "energy 0.3 somewhere else". Otherwise a planner would learn that dying pays (D5).
        The env state after the call is post-respawn, and the next tick starts from there."""
        actions = np.asarray(actions, dtype=np.int64)
        self.pos = (self.pos + MOVES[actions]) % self.n
        self.energy = self.energy - self.cfg.energy_decay
        # Both agents eat if they share a cell: keeps energy a deterministic function of the inputs.
        eaten = []
        for i in range(N_AGENTS):
            if self.food[self.pos[i, 0], self.pos[i, 1]]:
                self.energy[i] = min(self.cfg.energy_max, self.energy[i] + self.cfg.food_gain)
                eaten.append((self.pos[i, 0], self.pos[i, 1]))
        for c in eaten:
            self.food[c] = False
        died = self.energy <= 0.0
        outcome = np.stack([
            np.concatenate([pos_enc(self.pos[i], self.n), [np.float32(max(0.0, self.energy[i]))]])
            for i in range(N_AGENTS)
        ]).astype(np.float32)
        # Respawn happens "between frames": random cell, low energy (the penalty).
        for i in np.flatnonzero(died):
            self.pos[i] = self.rng.integers(0, self.n, size=2)
            self.energy[i] = self.cfg.energy_respawn
        if self.rng.random() < self.cfg.food_spawn_prob:
            self._spawn_one()
        self.last_action = actions.copy()
        return outcome, died
