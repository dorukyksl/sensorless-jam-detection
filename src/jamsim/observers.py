"""Sensorless speed and angle observers (Python reference).

Both observers run once per control period and work in the estimated rotor
frame (gamma, delta). They share one discretisation: the stator current model

    L di/dt = v - R i - j w L i - e        (complex form, i = i_gamma + j i_delta)

is stepped exactly over one period, assuming w and v constant. At this motor's
rated speed the rotor turns 18 to 36 electrical degrees per sample at
10 to 20 kHz, and a forward Euler step of the same model has a pole outside
the unit circle (|lambda| = 1.03 to 1.14), so Euler is not used.

The C port in c/src mirrors these functions line by line. Keep them in step.

State and parameter vectors are flat float64 arrays so the functions can be
called from the jitted simulator.
"""

import math

import numpy as np
from numba import njit

TWO_PI = 2.0 * math.pi

# MRAS
# Westin 2016 eq. 4.7 to 4.9: the motor is the reference model, the dq
# current model with estimated speed is the adjustable model, and a PI
# adaptation law drives the speed estimate.
M_ID, M_IQ, M_W, M_INT, M_TH = range(5)
MRAS_NSTATE = 5
MP_TS, MP_R, MP_L, MP_PSI, MP_KP, MP_KI = range(6)
MRAS_NPARAM = 6

# Back EMF observer + PLL
# Luenberger observer for the current model with the back EMF as a constant
# disturbance in the estimated frame, then a PLL that drives the gamma
# component of the estimated EMF to zero.
E_IG, E_ID, E_EG, E_ED, E_TH, E_WF, E_WI = range(7)
EMF_NSTATE = 7
EP_TS, EP_R, EP_L, EP_PSI, EP_L1, EP_L2, EP_KP, EP_KI, EP_EMIN = range(9)
EMF_NPARAM = 9


@njit(cache=True)
def wrap_angle(x):
    while x >= math.pi:
        x -= TWO_PI
    while x < -math.pi:
        x += TWO_PI
    return x


@njit(cache=True)
def propagate(zr, zi, ur, ui, w, Ts, R, L):
    """Exact step of dz/dt = (-R/L - j w) z + u/L over Ts with w and u held."""
    a = R / L
    alpha = math.exp(-a * Ts)
    phi = w * Ts
    c = math.cos(phi)
    s = math.sin(phi)
    er = alpha * c
    ei = -alpha * s
    nr = er - 1.0
    ni = ei
    den = a * a + w * w
    gr = (-a * nr - w * ni) / den
    gi = (w * nr - a * ni) / den
    ur = ur / L
    ui = ui / L
    zr2 = er * zr - ei * zi + gr * ur - gi * ui
    zi2 = er * zi + ei * zr + gr * ui + gi * ur
    return zr2, zi2


@njit(cache=True)
def park(xa, xb, th):
    c = math.cos(th)
    s = math.sin(th)
    return xa * c + xb * s, -xa * s + xb * c


# MRAS
def mras_params(Ts, R, L, psi, bw_hz, zeta=1.0):
    """PI gains from the linearised loop.

    Near steady state the adaptation signal behaves like
    eps ~ -(psi/L)^2 / (s + R/L) * (w_hat - w), so with a PI law the closed
    loop is s^2 + (R/L + G Kp) s + G Ki with G = (psi/L)^2. Pick wn, zeta.
    """
    G = (psi / L) ** 2
    wn = TWO_PI * bw_hz
    kp = max(2.0 * zeta * wn - R / L, 0.0) / G
    ki = wn * wn / G
    p = np.zeros(MRAS_NPARAM)
    p[MP_TS], p[MP_R], p[MP_L], p[MP_PSI], p[MP_KP], p[MP_KI] = Ts, R, L, psi, kp, ki
    return p


@njit(cache=True)
def mras_init(x, th, w, idq_d, idq_q):
    x[M_ID] = idq_d
    x[M_IQ] = idq_q
    x[M_W] = w
    x[M_INT] = w
    x[M_TH] = th


@njit(cache=True)
def mras_step(x, p, ia, ib, ua, ub):
    """One update. ia, ib: currents sampled now. ua, ub: alpha-beta voltage
    applied over the period that just ended. Returns (theta_hat, w_hat),
    electrical angle and speed, for use in this period."""
    Ts = p[MP_TS]
    R = p[MP_R]
    L = p[MP_L]
    psi = p[MP_PSI]
    w = x[M_W]
    th = x[M_TH]
    # voltage in the estimated frame at the middle of the last period
    ug, ud = park(ua, ub, th + 0.5 * w * Ts)
    # adjustable model: input v - j psi w
    zr, zi = propagate(x[M_ID], x[M_IQ], ug, ud - psi * w, w, Ts, R, L)
    th = wrap_angle(th + w * Ts)
    ig, idl = park(ia, ib, th)
    # Popov adaptation signal, Westin eq. (4.8)
    eps = (zi - idl) * (ig + psi / L) - (zr - ig) * idl
    x[M_INT] += p[MP_KI] * Ts * eps
    w = p[MP_KP] * eps + x[M_INT]
    x[M_ID] = zr
    x[M_IQ] = zi
    x[M_W] = w
    x[M_TH] = th
    return th, w


# Back EMF observer + PLL
def emf_params(Ts, R, L, psi, obs_bw_hz, pll_bw_hz, zeta=1.0, emin_frac=0.05,
               w_rated_e=None):
    """Observer: both error poles at exp(-2 pi obs_bw Ts) per axis.
    PLL: type 2, s^2 + Kp s + Ki with wn = 2 pi pll_bw.
    emin: floor on |e_hat| when normalising the phase error."""
    a = R / L
    alpha = math.exp(-a * Ts)
    beta = (1.0 - alpha) / R
    z = math.exp(-TWO_PI * obs_bw_hz * Ts)
    l1 = 1.0 - z * z / alpha
    l2 = (1.0 - z) * (1.0 - z) / beta
    wn = TWO_PI * pll_bw_hz
    if w_rated_e is None:
        w_rated_e = 8500 * math.pi / 30 * 7
    p = np.zeros(EMF_NPARAM)
    p[EP_TS], p[EP_R], p[EP_L], p[EP_PSI] = Ts, R, L, psi
    p[EP_L1], p[EP_L2] = l1, l2
    p[EP_KP], p[EP_KI] = 2.0 * zeta * wn, wn * wn
    p[EP_EMIN] = emin_frac * psi * w_rated_e
    return p


@njit(cache=True)
def emf_init(x, th, w, ig, idl, psi):
    x[E_IG] = ig
    x[E_ID] = idl
    x[E_EG] = 0.0
    x[E_ED] = psi * w
    x[E_TH] = th
    x[E_WF] = w
    x[E_WI] = w


@njit(cache=True)
def emf_step(x, p, ia, ib, ua, ub):
    """Same interface as mras_step. Returns (theta_hat, w_hat) with w_hat the
    PLL integrator, which is the smoother of the two PLL speed signals."""
    Ts = p[EP_TS]
    R = p[EP_R]
    L = p[EP_L]
    wf = x[E_WF]
    th = x[E_TH]
    ug, ud = park(ua, ub, th + 0.5 * wf * Ts)
    zr, zi = propagate(x[E_IG], x[E_ID], ug - x[E_EG], ud - x[E_ED], wf, Ts, R, L)
    th = wrap_angle(th + wf * Ts)
    ig, idl = park(ia, ib, th)
    rg = ig - zr
    rd = idl - zi
    l1 = p[EP_L1]
    l2 = p[EP_L2]
    x[E_IG] = zr + l1 * rg
    x[E_ID] = zi + l1 * rd
    eg = x[E_EG] - l2 * rg
    ed = x[E_ED] - l2 * rd
    x[E_EG] = eg
    x[E_ED] = ed
    # phase error: e_hat = |e| (-sin d, cos d) for angle error d = th - th_hat
    mag = math.sqrt(eg * eg + ed * ed)
    if mag < p[EP_EMIN]:
        mag = p[EP_EMIN]
    err = -eg / mag
    if ed < 0.0:
        err = -err
    x[E_WI] += p[EP_KI] * Ts * err
    x[E_WF] = p[EP_KP] * err + x[E_WI]
    x[E_TH] = th
    return th, x[E_WI]
