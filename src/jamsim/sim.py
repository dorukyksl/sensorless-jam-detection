"""Closed-loop simulation: PMSM plant, sampled FOC, observer, detector.

Timing per control period k (Ts = 1/fs):
  1. phase currents a and b are sampled at t = k Ts (noise + ADC rounding)
  2. the observer updates with these currents and with the voltage that was
     applied during the period that just ended
  3. the detector sees the new speed estimate
  4. speed and current PI compute a voltage; it is applied during the NEXT
     period (one period of computation delay), so the inverse Park uses
     theta_hat + 1.5 w_hat Ts
  5. the plant is integrated over [k Ts, (k+1) Ts) with RK4 and n_sub
     substeps, using the voltage computed one period earlier

The inverter is an average model limited to the inscribed circle of the
SVM hexagon, |v| <= Vdc/sqrt(3). No dead time and no switching ripple.

Load torque acts like friction, T_L = T(t) * tanh(w / w_slip), so a large
load stalls the rotor instead of driving it backwards. T(t) is the base
load plus a ripple locked to rotor angle.
"""

import math

import numpy as np
from numba import njit

from .params import (P_TS, P_NSUB, P_R, P_L, P_PSI, P_P, P_J, P_B, P_VDC,
                     P_IMAX, P_KPI, P_KII, P_KPW, P_KIW, P_WREF, P_NOISE,
                     P_LSB, P_WSLIP, P_RIPN, P_IBRAKE, P_WOFF)
from .observers import (mras_init, mras_step, emf_init, emf_step, park,
                        wrap_angle, MRAS_NSTATE, EMF_NSTATE)
from .detector import det_init, det_step, D_NMAX

OBS_IDEAL = 0
OBS_MRAS = 1
OBS_EMF = 2
OBS_NAMES = {OBS_IDEAL: "ideal sensor", OBS_MRAS: "MRAS", OBS_EMF: "EMF+PLL"}
NO_DETECT = 1e12   # threshold that never fires

SQRT3 = math.sqrt(3.0)


@njit(cache=True)
def _deriv(ia, ib, wm, thm, va, vb, tload, tripple, R, L, psi, p, J, B,
           wslip, ripn):
    the = p * thm
    s = math.sin(the)
    c = math.cos(the)
    we = p * wm
    dia = (va - R * ia + psi * we * s) / L
    dib = (vb - R * ib - psi * we * c) / L
    iq = -ia * s + ib * c
    te = 1.5 * p * psi * iq
    tl = (tload + tripple * math.sin(ripn * thm)) * math.tanh(wm / wslip)
    dwm = (te - B * wm - tl) / J
    return dia, dib, dwm, wm


@njit(cache=True)
def _adc(x, lsb, noise):
    y = x + noise * np.random.standard_normal()
    return lsb * math.floor(y / lsb + 0.5)


@njit(cache=True)
def simulate(P, obs_type, obs_p, tbase, tripple, seed, det_n, det_ath,
             det_wen, brake):
    """Run len(tbase)-1 control periods.

    Returns (wm_true, wm_hat, dtheta, iq, iq_ref, fired_k, obs_in).
    dtheta is the electrical angle error theta - theta_hat in rad. fired_k is
    the first period the detector fired (-1 if never). With brake=True the
    controller switches to braking at that period. obs_in (n x 4) holds the
    observer inputs ia, ib, ua, ub of every period, for the C test vectors.
    """
    np.random.seed(seed)
    Ts = P[P_TS]
    nsub = int(P[P_NSUB])
    R = P[P_R]
    L = P[P_L]
    psi = P[P_PSI]
    p = P[P_P]
    J = P[P_J]
    B = P[P_B]
    vmax = P[P_VDC] / SQRT3
    imax = P[P_IMAX]
    kpi = P[P_KPI]
    kii = P[P_KII]
    kpw = P[P_KPW]
    kiw = P[P_KIW]
    wref = P[P_WREF]
    noise = P[P_NOISE]
    lsb = P[P_LSB]
    wslip = P[P_WSLIP]
    ripn = P[P_RIPN]
    kt = 1.5 * p * psi

    n = tbase.shape[0] - 1
    out_w = np.zeros(n)
    out_wh = np.zeros(n)
    out_dth = np.zeros(n)
    out_iq = np.zeros(n)
    out_iqr = np.zeros(n)
    obs_in = np.zeros((n, 4))

    # steady state at rated speed with the initial load
    wm = wref
    thm = 0.0
    iq0 = (tbase[0] * math.tanh(wm / wslip) + B * wm) / kt
    ia = 0.0
    ib = iq0
    we = p * wm
    vd0 = -we * L * iq0
    vq0 = R * iq0 + we * psi
    thv = -0.5 * we * Ts
    ua = vd0 * math.cos(thv) - vq0 * math.sin(thv)   # applied during last period
    ub = vd0 * math.sin(thv) + vq0 * math.cos(thv)
    thv = 0.5 * we * Ts
    ua_next = vd0 * math.cos(thv) - vq0 * math.sin(thv)  # applied during period 0
    ub_next = vd0 * math.sin(thv) + vq0 * math.cos(thv)

    int_w = iq0
    int_d = 0.0
    int_q = R * iq0

    xm = np.zeros(MRAS_NSTATE)
    xe = np.zeros(EMF_NSTATE)
    if obs_type == OBS_MRAS:
        # observers hold the angle of the previous sample; their first step
        # advances it to theta(0) = 0
        mras_init(xm, -we * Ts, we, 0.0, iq0)
    elif obs_type == OBS_EMF:
        emf_init(xe, -we * Ts, we, 0.0, iq0, psi)
    det = np.zeros(D_NMAX + 3)
    det_init(det, wm)
    inv_nts = 1.0 / (det_n * Ts)

    mode = 0          # 0 run, 1 brake, 2 coast
    fired_k = -1
    h = Ts / nsub

    for k in range(n):
        # 1. sample
        the = p * thm
        i_a = _adc(ia, lsb, noise)
        i_b = _adc(-0.5 * ia + 0.5 * SQRT3 * ib, lsb, noise)
        ma = i_a
        mb = (i_a + 2.0 * i_b) / SQRT3

        # 2. observer (ua, ub = voltage applied over the period that ended)
        obs_in[k, 0] = ma
        obs_in[k, 1] = mb
        obs_in[k, 2] = ua
        obs_in[k, 3] = ub
        if obs_type == OBS_MRAS:
            th_h, we_h = mras_step(xm, obs_p, ma, mb, ua, ub)
        elif obs_type == OBS_EMF:
            th_h, we_h = emf_step(xe, obs_p, ma, mb, ua, ub)
        else:
            th_h = wrap_angle(the)
            we_h = p * wm
        wm_h = we_h / p

        out_w[k] = wm
        out_wh[k] = wm_h
        out_dth[k] = wrap_angle(the - th_h)

        # 3. detector
        if det_step(det, wm_h, det_n, inv_nts, det_ath, det_wen) == 1 and fired_k < 0:
            fired_k = k
            if brake:
                mode = 1
        if mode == 1 and wm_h < P[P_WOFF]:
            mode = 2

        # 4. control
        ig, idl = park(ma, mb, th_h)
        if mode == 0:
            ew = wref - wm_h
            iq_ref = kpw * ew + int_w
            if iq_ref > imax:
                iq_ref = imax
            elif iq_ref < -imax:
                iq_ref = -imax
            else:
                int_w += kiw * Ts * ew
        elif mode == 1:
            iq_ref = -P[P_IBRAKE]
        else:
            iq_ref = 0.0
        ed = 0.0 - ig
        eq = iq_ref - idl
        vd = kpi * ed + int_d - we_h * L * idl
        vq = kpi * eq + int_q + we_h * (psi + L * ig)
        vm = math.sqrt(vd * vd + vq * vq)
        if vm > vmax:
            vd *= vmax / vm
            vq *= vmax / vm
        else:
            int_d += kpi * kii * Ts * ed
            int_q += kpi * kii * Ts * eq
        thc = th_h + 1.5 * we_h * Ts
        c = math.cos(thc)
        s = math.sin(thc)
        out_iq[k] = idl
        out_iqr[k] = iq_ref

        # 5. plant over this period with the voltage computed last period
        va = ua_next
        vb = ub_next
        t0 = tbase[k]
        dt_l = (tbase[k + 1] - t0) / nsub
        r0 = tripple[k]
        dr_l = (tripple[k + 1] - r0) / nsub
        for j in range(nsub):
            tl0 = t0 + dt_l * j
            rp0 = r0 + dr_l * j
            tlm = tl0 + 0.5 * dt_l
            rpm_ = rp0 + 0.5 * dr_l
            tl1 = tl0 + dt_l
            rp1 = rp0 + dr_l
            k1a, k1b, k1w, k1t = _deriv(ia, ib, wm, thm, va, vb, tl0, rp0, R, L, psi, p, J, B, wslip, ripn)
            k2a, k2b, k2w, k2t = _deriv(ia + 0.5 * h * k1a, ib + 0.5 * h * k1b, wm + 0.5 * h * k1w,
                                        thm + 0.5 * h * k1t, va, vb, tlm, rpm_, R, L, psi, p, J, B, wslip, ripn)
            k3a, k3b, k3w, k3t = _deriv(ia + 0.5 * h * k2a, ib + 0.5 * h * k2b, wm + 0.5 * h * k2w,
                                        thm + 0.5 * h * k2t, va, vb, tlm, rpm_, R, L, psi, p, J, B, wslip, ripn)
            k4a, k4b, k4w, k4t = _deriv(ia + h * k3a, ib + h * k3b, wm + h * k3w,
                                        thm + h * k3t, va, vb, tl1, rp1, R, L, psi, p, J, B, wslip, ripn)
            ia += h / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
            ib += h / 6.0 * (k1b + 2 * k2b + 2 * k3b + k4b)
            wm += h / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
            thm += h / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        # thm is left unwrapped: the ripple term uses a non-integer
        # cycles-per-revolution factor and float64 keeps plenty of resolution

        ua = va
        ub = vb
        ua_next = vd * c - vq * s
        ub_next = vd * s + vq * c

    return out_w, out_wh, out_dth, out_iq, out_iqr, fired_k, obs_in
