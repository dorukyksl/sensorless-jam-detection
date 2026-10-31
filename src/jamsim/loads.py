"""Synthetic load torque profiles.

There is no public recording of chainsaw load torque, so every profile here
is synthetic. The shapes and ranges are my assumptions and are listed in the
README. Loads are given as two arrays sampled at the control rate:

    tbase[k]    base load torque magnitude [Nm]
    tripple[k]  amplitude of a ripple locked to rotor angle [Nm]

The plant turns them into a friction-like torque, see sim.py.
"""

from dataclasses import dataclass

import numpy as np

IDLE_NM = 0.05        # chain and bar friction with no cut (assumption)
RATED_NM = 1.5        # Westin 2016, Table 1.1


@dataclass
class Event:
    kind: str          # "jam" or "snag"
    k_on: int          # control period at which the event starts
    severity: float    # jam: final torque [Nm]; snag: added peak [Nm]
    shape_ms: float    # jam: rise time; snag: pulse length
    level: float       # base load just before the event [Nm]


def _lowpass_noise(rng, n, fs, fc, sigma):
    """First-order low-pass filtered white noise with std ~ sigma."""
    a = np.exp(-2 * np.pi * fc / fs)
    w = rng.standard_normal(n) * sigma * np.sqrt(1 - a * a)
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = a * acc + w[i]
        y[i] = acc
    return y


def _half_sine(n_len):
    return np.sin(np.pi * (np.arange(n_len) + 0.5) / n_len)


def normal_cutting(rng, fs, duration):
    """Repeated cuts: idle, entry ramp, held cut with noise, knots and
    ripple, exit ramp. 30 % of entries are hard steps (0.5 to 2 ms rise)."""
    n = int(round(duration * fs)) + 1
    tb = np.full(n, IDLE_NM)
    tr = np.zeros(n)
    k = int(0.15 * fs)
    while k < n:
        k += int(rng.uniform(0.1, 0.4) * fs)          # idle gap
        if k >= n:
            break
        level = rng.uniform(0.3, 1.4)
        hard = rng.random() < 0.3
        rise = rng.uniform(0.5e-3, 2e-3) if hard else rng.uniform(20e-3, 150e-3)
        hold = rng.uniform(0.3, 1.0)
        fall = rng.uniform(5e-3, 80e-3)
        nr = max(int(rise * fs), 1)
        nh = int(hold * fs)
        nf = max(int(fall * fs), 1)
        seg = np.concatenate([
            np.linspace(IDLE_NM, level, nr, endpoint=False),
            np.full(nh, level),
            np.linspace(level, IDLE_NM, nf, endpoint=False),
        ])
        held = slice(nr, nr + nh)
        seg[held] += _lowpass_noise(rng, nh, fs, 300.0, 0.1 * level)
        t_knot = 0.0
        while True:                                  # knots, 3 per second
            t_knot += rng.exponential(1 / 3.0)
            kk = int(t_knot * fs)
            if kk >= nh:
                break
            nl = max(int(rng.uniform(2e-3, 20e-3) * fs), 2)
            amp = rng.uniform(0.1, 0.5)
            end = min(kk + nl, nh)
            seg[nr + kk:nr + end] += amp * _half_sine(nl)[:end - kk]
        rip = np.zeros(seg.size)
        rip[held] = rng.uniform(0.05, 0.2) * level
        end = min(k + seg.size, n)
        tb[k:end] = seg[:end - k]
        tr[k:end] = rip[:end - k]
        k = end
    tr = np.minimum(tr, 0.2)
    tb = np.clip(tb, 0.0, RATED_NM - tr)
    return tb, tr


def event_run(rng, fs, kind, severity, pre=0.30, post=0.08):
    """Steady cut at a random level, then one event. Returns tb, tr, Event."""
    n = int(round((pre + post) * fs)) + 1
    level = rng.uniform(0.3, 1.2)
    tb = np.full(n, level)
    tb += _lowpass_noise(rng, n, fs, 300.0, 0.1 * level)
    tr = np.full(n, rng.uniform(0.05, 0.2) * level)
    k_on = int((pre - rng.uniform(0.0, 0.05)) * fs)
    if kind == "jam":
        shape = rng.uniform(0.5, 3.0)
        nr = max(int(shape * 1e-3 * fs), 1)
        ramp = np.linspace(tb[k_on], severity, nr + 1)[1:]
        tb[k_on:k_on + nr] = ramp
        tb[k_on + nr:] = severity
    elif kind == "snag":
        shape = rng.uniform(3.0, 15.0)
        nl = max(int(shape * 1e-3 * fs), 2)
        tb[k_on:k_on + nl] += severity * _half_sine(nl)
    else:
        raise ValueError(kind)
    tb = np.maximum(tb, 0.0)
    return tb, tr, Event(kind, k_on, severity, shape, level)
