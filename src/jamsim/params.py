"""Motor, inverter and controller parameters.

Electrical and mechanical motor data come from the public KTH master thesis
by I. Westin, "Sensorless Control of a PMSM: Evaluation of Different Speed
and Position Estimation Methods" (2016, diva2:1032918), which studies a
battery chainsaw motor: Table 1.1 and the MATLAB script in Appendix A.
Everything else (noise, ADC, controller tuning, inertia factor)
is my own assumption and is marked as such.
"""

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class Motor:
    # Westin 2016, Appendix A
    R: float = 0.0087            # stator resistance [ohm]
    L: float = 19e-6             # stator inductance, SPMSM so Ld = Lq [H]
    psi: float = 2.13 * math.sqrt(2) * 60 / (math.sqrt(3) * 1000 * 2 * math.pi * 7)  # PM flux [Vs/rad]
    p: int = 7                   # pole pairs (14 poles, Table 1.1)
    J_rotor: float = 0.3e-4      # rotor inertia [kg m^2]
    B: float = 0.05e-3           # viscous damping [Nm/(rad/s)]
    # Westin 2016, Table 1.1
    Vdc: float = 30.0            # DC link [V]
    Imax: float = 60.0           # maximum current [A]
    rated_rpm: float = 8500.0
    rated_torque: float = 1.5    # [Nm]

    @property
    def Kt(self) -> float:
        """Torque constant [Nm/A] for amplitude-invariant dq currents."""
        return 1.5 * self.p * self.psi


@dataclass(frozen=True)
class Setup:
    """Simulation setup. Fields without a thesis source are assumptions."""
    motor: Motor = field(default_factory=Motor)
    fs: float = 15e3             # control and sampling rate [Hz]
    n_sub: int = 8               # plant RK4 substeps per control period
    inertia_factor: float = 1.0  # total inertia / rotor inertia (chain and sprocket unknown)
    noise_A: float = 0.2         # current sensor noise, 1 sigma per phase [A] (assumption)
    adc_fullscale_A: float = 80.0  # bipolar ADC range [A] (assumption)
    adc_bits: int = 12
    speed_bw_hz: float = 85.0    # speed loop crossover [Hz]; gives the thesis droop of about 450 rpm at 1 Nm
    w_slip: float = 2.0          # load friction smoothing speed [rad/s]
    ripple_per_rev: float = 3.5  # load ripple cycles per mechanical revolution (assumption)

    @property
    def J(self) -> float:
        return self.motor.J_rotor * self.inertia_factor

    @property
    def Ts(self) -> float:
        return 1.0 / self.fs

    @property
    def w_ref(self) -> float:
        """Mechanical speed reference [rad/s]."""
        return self.motor.rated_rpm * math.pi / 30.0


# Index constants for the flat parameter vector handed to the jitted simulator.
P_TS, P_NSUB, P_R, P_L, P_PSI, P_P, P_J, P_B, P_VDC, P_IMAX, \
    P_KPI, P_KII, P_KPW, P_KIW, P_WREF, P_NOISE, P_LSB, P_WSLIP, P_RIPN, \
    P_IBRAKE, P_WOFF = range(21)
N_PARAMS = 21


def controller_gains(s: Setup):
    """Current and speed PI gains.

    Current loop: series PI as in Westin 2016 eq. 4.1 and 4.2, Kp = L*BWc and
    Ki = R/L, with BWc = 2*pi*fs/20 (the thesis uses 2*pi*15000/20).
    Speed loop: parallel PI with crossover speed_bw_hz and the PI zero a
    quarter of the way down. The Appendix A speed gains do not reproduce the
    thesis' own Figure 4.3, so the speed loop is tuned here instead.
    """
    m = s.motor
    bwc = 2 * math.pi * s.fs / 20.0
    kp_i = m.L * bwc
    ki_i = m.R / m.L
    wc = 2 * math.pi * s.speed_bw_hz
    kp_w = s.J * wc / m.Kt
    ki_w = kp_w * wc / 4.0
    return kp_i, ki_i, kp_w, ki_w


def param_vector(s: Setup, i_brake: float = 60.0, w_off_rpm: float = 1000.0) -> np.ndarray:
    m = s.motor
    kp_i, ki_i, kp_w, ki_w = controller_gains(s)
    v = np.zeros(N_PARAMS)
    v[P_TS] = s.Ts
    v[P_NSUB] = s.n_sub
    v[P_R] = m.R
    v[P_L] = m.L
    v[P_PSI] = m.psi
    v[P_P] = m.p
    v[P_J] = s.J
    v[P_B] = m.B
    v[P_VDC] = m.Vdc
    v[P_IMAX] = m.Imax
    v[P_KPI] = kp_i
    v[P_KII] = ki_i
    v[P_KPW] = kp_w
    v[P_KIW] = ki_w
    v[P_WREF] = s.w_ref
    v[P_NOISE] = s.noise_A
    v[P_LSB] = 2 * s.adc_fullscale_A / (2 ** s.adc_bits)
    v[P_WSLIP] = s.w_slip
    v[P_RIPN] = s.ripple_per_rev
    v[P_IBRAKE] = i_brake
    v[P_WOFF] = w_off_rpm * math.pi / 30.0
    return v
