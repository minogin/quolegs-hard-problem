"""Parallel batch runner with on-disk caching of RunResults."""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

from ..sim import run_simulation


def _init():
    torch.set_num_threads(1)


def _worker(args):
    rc, wirings, meta = args
    return run_simulation(rc, wirings, meta)


def _call(fn_args):
    fn, kwargs = fn_args
    return fn(**kwargs)


def run_batch(jobs, workers=None, cache_dir=None, label="batch"):
    """jobs: list of (RunConfig, wirings|None, meta dict). Returns list of RunResult (ordered)."""
    results = [None] * len(jobs)
    todo = []
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for i, (rc, _, _) in enumerate(jobs):
            p = os.path.join(cache_dir, f"seed{rc.seed}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    results[i] = pickle.load(f)
            else:
                todo.append(i)
    else:
        todo = list(range(len(jobs)))
    if todo:
        t0 = time.time()
        workers = workers or max(1, os.cpu_count() - 1)
        print(f"[{label}] running {len(todo)} sims on {workers} workers", flush=True)
        with Pool(workers, initializer=_init) as pool:
            for k, (i, res) in enumerate(zip(todo, pool.imap(_worker, [jobs[i] for i in todo]))):
                results[i] = res
                if cache_dir:
                    with open(os.path.join(cache_dir, f"seed{jobs[i][0].seed}.pkl"), "wb") as f:
                        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
                if (k + 1) % max(1, len(todo) // 10) == 0 or k + 1 == len(todo):
                    el = time.time() - t0
                    print(f"[{label}] {k + 1}/{len(todo)} done, {el:.0f}s elapsed, "
                          f"eta {el / (k + 1) * (len(todo) - k - 1):.0f}s", flush=True)
    return results


def run_parallel(fn, kwargs_list, workers=None, label="jobs"):
    """Run an arbitrary picklable function over kwargs in parallel."""
    workers = workers or max(1, os.cpu_count() - 1)
    t0 = time.time()
    print(f"[{label}] {len(kwargs_list)} jobs on {workers} workers", flush=True)
    out = []
    with Pool(workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(_call, [(fn, kw) for kw in kwargs_list])):
            out.append(r)
            print(f"[{label}] {k + 1}/{len(kwargs_list)} done, {time.time() - t0:.0f}s", flush=True)
    return out


def late(x, frac=0.25):
    n = len(x)
    return x[int(n * (1 - frac)):]
