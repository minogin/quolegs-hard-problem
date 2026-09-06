"""Wiring config: the ONLY place where S and O differ. Models, planner and env never read role names."""
import copy

SOURCES = ("internal_state", "perceived_other")

DEFAULT_WIRING = {
    "S": {"input": "internal_state", "target": "internal_state", "drives_planner": True},
    "O": {"input": "perceived_other", "target": "perceived_other", "drives_planner": False},
}


def validate(w: dict):
    drivers = [k for k, v in w.items() if v["drives_planner"]]
    assert len(drivers) == 1, f"exactly one model must drive the planner, got {drivers}"
    for k, v in w.items():
        assert v["input"] in SOURCES and v["target"] in SOURCES, (k, v)
    return w


def swapped(w: dict) -> dict:
    """Swap the wiring entries of the two agent models (names stay, roles move)."""
    keys = list(w.keys())
    assert len(keys) == 2
    out = copy.deepcopy(w)
    out[keys[0]], out[keys[1]] = copy.deepcopy(w[keys[1]]), copy.deepcopy(w[keys[0]])
    return validate(out)


def driver(w: dict) -> str:
    return [k for k, v in w.items() if v["drives_planner"]][0]


def rewire(w: dict, name: str, **entry) -> dict:
    out = copy.deepcopy(w)
    out[name].update(entry)
    return validate(out)


def set_driver(w: dict, name: str) -> dict:
    out = copy.deepcopy(w)
    for k in out:
        out[k]["drives_planner"] = (k == name)
    return validate(out)
