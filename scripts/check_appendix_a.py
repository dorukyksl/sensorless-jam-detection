"""Run the continuous FOC model of Westin 2016, Appendix A, as printed.

The thesis (Figure 4.3) reports 2.9 % overshoot on the step to 8500 rpm and
at most 450 rpm droop under a 1 Nm step load. With the speed gains printed in
Appendix A (delta = 11, tau = 1/200) this script does not reproduce that,
which is why the speed loop in this repository is tuned separately
(see Setup.speed_bw_hz). The PI integrators are written as ODE states
instead of the persistent variables of the original script.

    python scripts/check_appendix_a.py
"""

import math

import numpy as np
from scipy.integrate import solve_ivp

R = 0.0087
Ls = 19e-6
ke = 2.13 * math.sqrt(2)
fe = ke * 60 / (math.sqrt(3) * 1000 * 2 * math.pi * 7)
p = 7
J = 0.3e-4
B = 0.05e-3
Vmax = 30 / math.sqrt(3)
Imax = 60
delta = 11
tao = 1 / 200
K = 1.5 * p * fe / J
Kp = Ls * 2 * math.pi * 15000 / 20
Ki = R / Ls
spdKp = 1 / (delta * K * tao)
spdKi = 1 / (delta ** 2 * tao)
w_ref = 8500 * math.pi / 30


def load(t):
    return 1.0 if 1.5 <= t < 2.5 else 0.0


def rhs(t, x):
    i_d, i_q, w, e_w, e_d, e_q = x
    iq_ref = min(max(spdKp * (w_ref - w) + spdKi * spdKp * e_w, -Imax), Imax)
    vd = min(max(-Ls * i_q * w * p + Kp * (0 - i_d) + Kp * Ki * e_d, -Vmax), Vmax)
    vq = min(max(w * p * (fe + Ls * i_d) + Kp * (iq_ref - i_q) + Ki * Kp * e_q, -Vmax), Vmax)
    return [(-R * i_d + vd + Ls * i_q * w * p) / Ls,
            (-R * i_q + vq - Ls * i_d * w * p - fe * w * p) / Ls,
            (1.5 * p * fe * i_q - B * w - load(t)) / J,
            w_ref - w, -i_d, iq_ref - i_q]


def main():
    sol = solve_ivp(rhs, (0, 3), [0] * 6, method="LSODA", max_step=50e-6, rtol=1e-6, atol=1e-6)
    rpm = sol.y[2] * 30 / math.pi
    loaded = (sol.t > 1.5) & (sol.t < 2.5)
    print(f"speed gains: Kp = {spdKp:.4f} A/(rad/s), series Ki = {spdKi:.3f} 1/s")
    print(f"overshoot on the step to 8500 rpm: {100 * (rpm.max() / 8500 - 1):.1f} % (thesis: 2.9 %)")
    print(f"lowest speed under the 1 Nm load: {rpm[loaded].min():.0f} rpm "
          f"(thesis: droop of at most 450 rpm, so about 8050 rpm)")


if __name__ == "__main__":
    main()
