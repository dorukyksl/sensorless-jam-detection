"""Write golden test vectors for the C port.

Each file holds a closed-loop run of the Python reference: the observer
inputs of every control period (sampled currents and applied voltage), the
Python observer outputs and the Python detector flag. The C tests replay the
inputs through the float32 C code and compare.

    python scripts/export_vectors.py            # writes c/vectors/*.csv
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamsim.params import Setup, param_vector                    # noqa: E402
from jamsim.sim import simulate, OBS_MRAS, OBS_EMF, NO_DETECT    # noqa: E402
from jamsim import observers as ob                               # noqa: E402
from jamsim.detector import det_init, det_step, D_NMAX           # noqa: E402
from jamsim.loads import event_run                               # noqa: E402

FS = 15e3
DET_N = 30                 # 2 ms window at 15 kHz
DET_ATH = 35000.0          # rad/s^2, fires on the jam, not on the 1.2 Nm step
OBS_BW = 400.0


def scenarios(fs):
    rng = np.random.default_rng(7)
    tb, tr, ev = event_run(rng, fs, "jam", 3.0, pre=0.10, post=0.04)
    yield "jam", tb, tr
    n = int(0.15 * fs) + 1
    tb = np.full(n, 0.2)
    k = int(0.05 * fs)
    nr = int(0.5e-3 * fs)
    tb[k:k + nr] = np.linspace(0.2, 1.4, nr)
    tb[k + nr:] = 1.4
    yield "step", tb, np.full(n, 0.1)


def replay(kind, op, obs_in, th0, w0, iq0, psi):
    """Run the Python observer open loop over recorded inputs."""
    n = obs_in.shape[0]
    th = np.zeros(n)
    w = np.zeros(n)
    if kind == OBS_MRAS:
        x = np.zeros(ob.MRAS_NSTATE)
        ob.mras_init(x, th0, w0, 0.0, iq0)
        step = ob.mras_step
    else:
        x = np.zeros(ob.EMF_NSTATE)
        ob.emf_init(x, th0, w0, 0.0, iq0, psi)
        step = ob.emf_step
    for k in range(n):
        th[k], w[k] = step(x, op, *obs_in[k])
    return th, w


def main(out=None):
    s = Setup(fs=FS)
    m = s.motor
    P = param_vector(s)
    Ts = s.Ts
    out = Path(out) if out is not None else ROOT / "c" / "vectors"
    out.mkdir(parents=True, exist_ok=True)
    observers = {
        "mras": (OBS_MRAS, ob.mras_params(Ts, m.R, m.L, m.psi, OBS_BW)),
        "emf": (OBS_EMF, ob.emf_params(Ts, m.R, m.L, m.psi, 4 * OBS_BW, OBS_BW)),
    }
    w_enable = 0.1 * s.w_ref
    for scen, tb, tr in scenarios(FS):
        for name, (kind, op) in observers.items():
            res = simulate(P, kind, op, tb, tr, 3, DET_N, NO_DETECT, w_enable, False)
            wm, wm_hat, obs_in = res[0], res[1], res[6]
            we0 = m.p * s.w_ref
            iq0 = (tb[0] * math.tanh(s.w_ref / s.w_slip) + m.B * s.w_ref) / m.Kt
            th0 = -we0 * Ts
            th, w = replay(kind, op, obs_in, th0, we0, iq0, m.psi)
            assert np.allclose(w / m.p, wm_hat), "replay differs from closed loop"
            det = np.zeros(D_NMAX + 3)
            det_init(det, w[0] / m.p)
            inv = 1.0 / (DET_N * Ts)
            fired = np.array([det_step(det, wk / m.p, DET_N, inv, DET_ATH, w_enable)
                              for wk in w])
            path = out / f"{name}_{scen}.csv"
            with open(path, "w") as f:
                f.write(f"# observer={name}\n# scenario={scen}\n")
                f.write(f"# Ts={Ts!r}\n# R={m.R!r}\n# L={m.L!r}\n# psi={m.psi!r}\n# p={m.p}\n")
                for i, v in enumerate(op):
                    f.write(f"# op{i}={float(v)!r}\n")
                f.write(f"# th0={th0!r}\n# w0={we0!r}\n# iq0={float(iq0)!r}\n")
                f.write(f"# det_n={DET_N}\n# det_ath={DET_ATH!r}\n# det_wen={w_enable!r}\n")
                f.write("# columns: ia ib ua ub th_ref w_ref fired_ref\n")
                for k in range(len(w)):
                    ia, ib, ua, ub = obs_in[k]
                    f.write(f"{ia:.9g},{ib:.9g},{ua:.9g},{ub:.9g},{th[k]:.9g},{w[k]:.9g},{fired[k]}\n")
            first = int(np.argmax(fired)) if fired.any() else -1
            print(f"{path.name}: {len(w)} rows, detector fires at row {first}")


if __name__ == "__main__":
    main()
