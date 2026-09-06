"""Predictive models: one MLP class for W, S and O. Knows nothing about roles.

A model here is "a net + its optimizer + a replay buffer". The only difference between W, S and O
is the input/output size; everything else (architecture, learning rule, buffer) is identical.
Design notes: docs/decisions.md (D9)."""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from .config import ModelConfig


class MLP(nn.Module):
    """input -> hidden -> hidden -> output, ReLU between layers ("MLP, 2 hidden layers" in the spec)."""

    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))


class Predictor:
    """MLP + Adam + replay buffer + online error / update-magnitude logging.

    Per environment tick the agent calls, in this order: error(x, y) -> push(x, y) -> train_step().
    The error is therefore measured *before* the model has seen the example (honest prediction error).
    Training samples random minibatches from the buffer rather than the latest example, so the model
    does not forget the past; S and O use exactly the same rule (D9)."""

    def __init__(self, in_dim: int, out_dim: int, cfg: ModelConfig, seed: int):
        self.in_dim, self.out_dim, self.cfg = in_dim, out_dim, cfg
        self.reset_params(seed)
        self.rng = np.random.default_rng(seed)
        # Ring buffer of (input, target) pairs: `ptr` is the write position, `size` how much is filled.
        self.bx = np.zeros((cfg.buffer, in_dim), dtype=np.float32)
        self.by = np.zeros((cfg.buffer, out_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def reset_params(self, seed: int):
        """Fresh random weights and a fresh optimizer (used by E1b's "fresh O" variant)."""
        torch.manual_seed(seed)
        self.net = MLP(self.in_dim, self.out_dim, self.cfg.hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    def clear_buffer(self):
        self.size = 0
        self.ptr = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass without learning. numpy in, numpy out; the planner calls this in batches."""
        with torch.inference_mode():
            return self.net(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()

    def error(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error of the prediction for one example. This is the logged "prediction error"."""
        p = self.predict(x[None])[0]
        return float(np.mean((p - y) ** 2))

    def push(self, x: np.ndarray, y: np.ndarray):
        self.bx[self.ptr] = x
        self.by[self.ptr] = y
        self.ptr = (self.ptr + 1) % self.cfg.buffer
        self.size = min(self.size + 1, self.cfg.buffer)

    def train_step(self) -> float:
        """One Adam step on a random replay minibatch. Returns ||delta theta||, the logged "update size"."""
        if self.size < self.cfg.batch:
            return 0.0
        idx = self.rng.integers(0, self.size, self.cfg.batch)
        x = torch.from_numpy(self.bx[idx])
        y = torch.from_numpy(self.by[idx])
        before = parameters_to_vector(self.net.parameters()).detach().clone()
        self.opt.zero_grad(set_to_none=True)
        loss = torch.mean((self.net(x) - y) ** 2)
        loss.backward()
        self.opt.step()
        after = parameters_to_vector(self.net.parameters()).detach()
        return float(torch.linalg.norm(after - before))

    def params_numpy(self):
        """Weights and biases in layer order, as plain numpy arrays.

        This is the form in which a model crosses the introspection boundary: no name, no optimizer,
        no buffer, just six tensors."""
        return [p.detach().numpy().copy() for p in (self.net.l1.weight, self.net.l1.bias,
                                                    self.net.l2.weight, self.net.l2.bias,
                                                    self.net.l3.weight, self.net.l3.bias)]

    def load_params(self, params):
        with torch.no_grad():
            for p, q in zip((self.net.l1.weight, self.net.l1.bias, self.net.l2.weight,
                             self.net.l2.bias, self.net.l3.weight, self.net.l3.bias), params):
                p.copy_(torch.from_numpy(np.asarray(q, dtype=np.float32)))

    def buffer_sample(self, n: int):
        """Random slice of the buffer; used to check that the S and O training streams match."""
        if self.size == 0:
            return None
        idx = self.rng.choice(self.size, size=min(n, self.size), replace=False)
        return self.bx[idx].copy(), self.by[idx].copy()


def mlp_forward_numpy(params, x):
    """The same net evaluated in pure numpy from a params list [W1,b1,W2,b2,W3,b3].

    The introspection layer uses this to compute outputs and hidden activations from raw weight arrays.
    Returns (output, h1, h2); the hidden activations feed the content features for E2."""
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.maximum(0.0, x @ W1.T + b1)
    h2 = np.maximum(0.0, h1 @ W2.T + b2)
    return h2 @ W3.T + b3, h1, h2
