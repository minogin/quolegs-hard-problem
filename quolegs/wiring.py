"""Wiring config: the ONLY place where S and O differ.

The model names ("S", "O") are just dictionary keys. Nothing in the agent, the models or the
introspection layer decides anything by these letters; they only read the three fields below.
Swapping the entries (see `swapped`) must swap the roles without any code change."""
import copy

# The two data streams an agent receives. Both have the same format and quality; sim.py builds them.
SOURCES = ("internal_state", "perceived_other")

DEFAULT_WIRING = {
    #        which stream feeds it   which stream it must predict   does the planner consult it
    "S": {"input": "internal_state", "target": "internal_state", "drives_planner": True},
    "O": {"input": "perceived_other", "target": "perceived_other", "drives_planner": False},
}


def validate(w: dict):
    """Exactly one model drives the planner; stream names must be known. Called on every change."""
    drivers = [k for k, v in w.items() if v["drives_planner"]]
    assert len(drivers) == 1, f"exactly one model must drive the planner, got {drivers}"
    for k, v in w.items():
        assert v["input"] in SOURCES and v["target"] in SOURCES, (k, v)
    return w


def swapped(w: dict) -> dict:
    """Swap the wiring entries of the two agent models: names stay, roles move (symmetry test)."""
    keys = list(w.keys())
    assert len(keys) == 2
    out = copy.deepcopy(w)
    out[keys[0]], out[keys[1]] = copy.deepcopy(w[keys[1]]), copy.deepcopy(w[keys[0]])
    return validate(out)


def driver(w: dict) -> str:
    return [k for k, v in w.items() if v["drives_planner"]][0]


def rewire(w: dict, name: str, **entry) -> dict:
    """Change what a model is fed / trained on (E1b). The model itself will drift to match."""
    out = copy.deepcopy(w)
    out[name].update(entry)
    return validate(out)


def set_driver(w: dict, name: str) -> dict:
    """Move the planner pointer only (E1a). Streams untouched: the model stays what it was."""
    out = copy.deepcopy(w)
    for k in out:
        out[k]["drives_planner"] = (k == name)
    return validate(out)
