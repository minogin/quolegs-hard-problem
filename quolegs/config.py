from dataclasses import dataclass, field, asdict, replace


@dataclass
class EnvConfig:
    n: int = 16                  # torus side
    k: int = 5                   # local window side
    food_spawn_prob: float = 0.25
    max_food: int = 12
    energy_init: float = 1.0
    energy_respawn: float = 0.3  # penalty: respawn with low energy
    energy_decay: float = 0.02
    food_gain: float = 0.3
    energy_max: float = 1.0


@dataclass
class ModelConfig:
    hidden: int = 64
    lr: float = 1e-3
    batch: int = 64
    buffer: int = 4000


@dataclass
class AgentConfig:
    horizon: int = 3
    gamma: float = 1.0
    epsilon: float = 0.1
    model: ModelConfig = field(default_factory=ModelConfig)
    self_noise: float = 0.0      # E5: gaussian noise on the internal_state stream (interoception)
    use_other_in_planning: bool = False  # D13: planner discounts food the other agent is predicted to reach first


@dataclass
class RunConfig:
    seed: int = 0
    steps: int = 4000
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_dict(self):
        return asdict(self)


def with_updates(cfg: RunConfig, **kw) -> RunConfig:
    return replace(cfg, **kw)
