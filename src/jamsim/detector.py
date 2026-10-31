"""Jam / snag detector on the estimated rotor speed.

Rule: the mean deceleration over the last N samples,

    a[k] = (w[k-N] - w[k]) / (N Ts)       (w = mechanical speed estimate)

must exceed a threshold A while w[k] is above an enable speed. The window
length N is the one design choice besides A: N = 1 is a plain threshold on
deceleration, a longer window asks for the deceleration to last.

detect_offline() and DetectorStep give the same trigger index; the C port
(c/src/detector.c) is tested against both.
"""

import math

import numpy as np
from numba import njit


def window_decel(w, n, Ts):
    """a[k] for every k >= n (NaN before)."""
    a = np.full(w.shape, np.nan)
    a[n:] = (w[:-n] - w[n:]) / (n * Ts)
    return a


def detect_offline(w, n, Ts, a_th, w_enable, start=0):
    """First index k >= start where the rule fires, or -1."""
    a = window_decel(w, n, Ts)
    ok = (a > a_th) & (w > w_enable)
    ok[:max(start, n)] = False
    idx = np.flatnonzero(ok)
    return int(idx[0]) if idx.size else -1


def peak_decel(w, n, Ts, w_enable, start=0, stop=None):
    """Largest a[k] with the enable condition, over [start, stop)."""
    a = window_decel(w, n, Ts)
    m = w > w_enable
    m[:max(start, n)] = False
    if stop is not None:
        m[stop:] = False
    if not m.any():
        return -math.inf
    return float(np.max(a[m]))


# In-loop version: state = [ring buffer (NMAX), head, count, fired]
D_NMAX = 256


@njit(cache=True)
def det_init(st, w0):
    for i in range(D_NMAX):
        st[i] = w0
    st[D_NMAX] = 0.0      # head
    st[D_NMAX + 1] = 0.0  # samples seen
    st[D_NMAX + 2] = 0.0  # fired flag


@njit(cache=True)
def det_step(st, w, n, inv_nts, a_th, w_enable):
    """Push w, return 1 if the rule fires on this sample (latched after)."""
    head = int(st[D_NMAX])
    seen = st[D_NMAX + 1]
    # oldest sample of the window sits n positions behind the write head
    old = st[(head - n) % D_NMAX] if seen >= n else w
    st[head] = w
    st[D_NMAX] = (head + 1) % D_NMAX
    st[D_NMAX + 1] = seen + 1.0
    if st[D_NMAX + 2] > 0.5:
        return 1
    if seen >= n and w > w_enable and (old - w) * inv_nts > a_th:
        st[D_NMAX + 2] = 1.0
        return 1
    return 0
