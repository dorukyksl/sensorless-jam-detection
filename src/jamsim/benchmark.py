"""Benchmark protocol.

For every sampling rate and observer:
  1. simulate set A (normal cutting, tuning) and set B (normal cutting,
     held out) and two event sets: T (tuning) and E (evaluation)
  2. for every observer bandwidth and detector window, set the threshold to
     1.1 x the largest windowed deceleration seen on set A
  3. pick the (bandwidth, window) pair with the highest detection rate on
     set T, ties broken by median latency
  4. report false alarms on set B and detection rate and latency on set E

The detector is evaluated offline on the logged speed estimate. That is
exact, because the run is identical up to the moment the detector fires.

An event counts as detected if the detector fires after the event starts
and before
  jam:  the true speed falls below the validity floor (20 % of rated), where
        a back EMF based estimate is no longer trustworthy
  snag: the end of the torque pulse + 5 ms
"""

from dataclasses import dataclass
import math

import numpy as np

from .params import Setup, param_vector
from .sim import simulate, OBS_IDEAL, OBS_MRAS, OBS_EMF, NO_DETECT
from .observers import mras_params, emf_params
from .loads import normal_cutting, event_run
from .detector import window_decel

MARGIN = 1.1
FLOOR = 0.2                      # validity floor and detector enable, x rated
WINDOWS_MS = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)   # 0.0 means one sample
BANDWIDTHS = (200.0, 400.0, 800.0)
JAM_LEVELS = (2.0, 2.5, 3.0, 4.0, 6.0)   # final load torque [Nm]
SNAG_LEVELS = (1.0, 2.0, 3.0, 4.0, 6.0)  # added peak torque [Nm]


@dataclass
class Protocol:
    n_normal: int = 40           # runs per normal set
    normal_s: float = 4.0        # seconds per normal run
    n_tune: int = 10             # events per level in set T
    n_eval: int = 40             # events per level in set E


def window_n(ms, fs):
    return max(1, int(round(ms * 1e-3 * fs)))


def observer_params(kind, bw, s):
    m = s.motor
    if kind == OBS_MRAS:
        return mras_params(s.Ts, m.R, m.L, m.psi, bw)
    if kind == OBS_EMF:
        return emf_params(s.Ts, m.R, m.L, m.psi, 4.0 * bw, bw)
    return np.zeros(1)


def run_normal(s, kind, op, seeds, duration):
    P = param_vector(s)
    out = []
    for seed in seeds:
        tb, tr = normal_cutting(np.random.default_rng(seed), s.fs, duration)
        w, wh = simulate(P, kind, op, tb, tr, seed, 1, NO_DETECT, 0.0, False)[:2]
        out.append((w, wh))
    return out


def run_events(s, kind, op, seed0, n_per_level):
    P = param_vector(s)
    out = []
    for k_ind, (name, levels) in enumerate((("jam", JAM_LEVELS), ("snag", SNAG_LEVELS))):
        for lev in levels:
            for j in range(n_per_level):
                seed = seed0 + 100000 * k_ind + 1000 * int(lev * 10) + j
                tb, tr, ev = event_run(np.random.default_rng(seed), s.fs, name, lev)
                w, wh = simulate(P, kind, op, tb, tr, seed, 1, NO_DETECT, 0.0, False)[:2]
                out.append((ev, w, wh))
    return out


def peak_over(runs, n, Ts, w_en, skip):
    pk = -math.inf
    for w, wh in runs:
        a = window_decel(wh, n, Ts)
        m = wh > w_en
        m[:max(skip, n)] = False
        if m.any():
            pk = max(pk, float(np.max(a[m])))
    return pk


def first_fire(wh, n, Ts, a_th, w_en, start):
    a = window_decel(wh, n, Ts)
    ok = (a > a_th) & (wh > w_en)
    ok[:max(start, n)] = False
    idx = np.flatnonzero(ok)
    return int(idx[0]) if idx.size else -1


def score_events(events, n, Ts, a_th, w_en, fs, w_floor):
    """Per event: (kind, severity, detected, latency_ms, false_before)."""
    rows = []
    skip = int(0.05 * fs)
    for ev, w, wh in events:
        k = first_fire(wh, n, Ts, a_th, w_en, skip)
        if ev.kind == "jam":
            below = np.flatnonzero((w < w_floor) & (np.arange(w.size) >= ev.k_on))
            deadline = int(below[0]) if below.size else w.size
        else:
            deadline = ev.k_on + int((ev.shape_ms + 5.0) * 1e-3 * fs)
        early = 0 <= k < ev.k_on
        hit = (k >= ev.k_on) and (k < deadline)
        lat = (k - ev.k_on) / fs * 1e3 if hit else math.nan
        rows.append((ev.kind, ev.severity, hit, lat, early))
    return rows


def count_false(runs, n, Ts, a_th, w_en, fs):
    skip = int(0.05 * fs)
    return sum(first_fire(wh, n, Ts, a_th, w_en, skip) >= 0 for _, wh in runs)


def evaluate_config(fs, kind, proto: Protocol, seed_base=0):
    """Full protocol for one sampling rate and observer family."""
    s = Setup(fs=fs)
    Ts = s.Ts
    w_floor = FLOOR * s.w_ref
    seeds_a = [seed_base + 10000 + i for i in range(proto.n_normal)]
    seeds_b = [seed_base + 20000 + i for i in range(proto.n_normal)]
    bws = (None,) if kind == OBS_IDEAL else BANDWIDTHS
    skip = int(0.05 * fs)
    cands = []
    cache = {}
    for bw in bws:
        op = observer_params(kind, bw, s)
        set_a = run_normal(s, kind, op, seeds_a, proto.normal_s)
        set_t = run_events(s, kind, op, seed_base + 300000, proto.n_tune)
        cache[bw] = (op, set_a)
        for ms in WINDOWS_MS:
            n = window_n(ms, fs)
            a_th = MARGIN * peak_over(set_a, n, Ts, w_floor, skip)
            rows = score_events(set_t, n, Ts, a_th, w_floor, fs, w_floor)
            rate = float(np.mean([r[2] for r in rows]))
            lats = [r[3] for r in rows if r[2]]
            med = float(np.median(lats)) if lats else math.inf
            cands.append(dict(bw=bw, window_ms=ms, n=n, a_th=a_th, tune_rate=rate,
                              tune_median_ms=med))

    def pick(cs):
        return sorted(cs, key=lambda c: (-c["tune_rate"], c["tune_median_ms"]))[0]

    chosen = pick(cands)
    plain = pick([c for c in cands if c["window_ms"] == 0.0])
    result = {"fs": fs, "observer": kind, "candidates": cands}
    for label, c in (("chosen", chosen), ("plain", plain)):
        op = cache[c["bw"]][0]
        set_b = run_normal(s, kind, op, seeds_b, proto.normal_s)
        set_e = run_events(s, kind, op, seed_base + 500000, proto.n_eval)
        rows = score_events(set_e, c["n"], Ts, c["a_th"], w_floor, fs, w_floor)
        fa = count_false(set_b, c["n"], Ts, c["a_th"], w_floor, fs)
        result[label] = dict(c, false_alarm_runs=int(fa), normal_runs=proto.n_normal,
                             normal_seconds=proto.n_normal * proto.normal_s,
                             events=[dict(kind=r[0], severity=r[1], detected=bool(r[2]),
                                          latency_ms=r[3], fired_early=bool(r[4]))
                                     for r in rows])
    return result


def summarise(events):
    """Detection rate and latency per (kind, severity)."""
    out = {}
    for kind in ("jam", "snag"):
        for sev in sorted({e["severity"] for e in events if e["kind"] == kind}):
            sel = [e for e in events if e["kind"] == kind and e["severity"] == sev]
            lats = [e["latency_ms"] for e in sel if e["detected"]]
            out[(kind, sev)] = dict(
                rate=float(np.mean([e["detected"] for e in sel])),
                median_ms=float(np.median(lats)) if lats else math.nan,
                p95_ms=float(np.percentile(lats, 95)) if lats else math.nan,
                n=len(sel))
    return out
