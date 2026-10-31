import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jamsim.params import Motor, Setup, param_vector                      # noqa: E402
from jamsim.observers import propagate, mras_params, emf_params           # noqa: E402
from jamsim.sim import simulate, OBS_IDEAL, OBS_MRAS, OBS_EMF, NO_DETECT  # noqa: E402
from jamsim.detector import det_init, det_step, detect_offline, D_NMAX    # noqa: E402

M = Motor()


def test_thesis_motor_is_consistent():
    # Kt * Imax should land on the rated torque of Table 1.1
    assert M.Kt * M.Imax == pytest.approx(M.rated_torque, rel=0.01)


@pytest.mark.parametrize("fs", [10e3, 15e3, 20e3])
def test_forward_euler_is_unstable_and_exact_step_is_not(fs):
    Ts = 1 / fs
    we = M.p * M.rated_rpm * math.pi / 30
    a = M.R / M.L
    euler = abs(complex(1 - a * Ts, -we * Ts))
    assert euler > 1.0
    zr, zi = propagate(1.0, 0.0, 0.0, 0.0, we, Ts, M.R, M.L)
    assert abs(complex(zr, zi)) == pytest.approx(math.exp(-a * Ts))


def test_propagate_matches_matrix_exponential():
    rng = np.random.default_rng(0)
    Ts = 1 / 15e3
    a = M.R / M.L
    for _ in range(20):
        w = rng.uniform(-9000, 9000)
        z = rng.normal(size=2) * 30
        u = rng.normal(size=2) * 10
        A = np.array([[-a, w], [-w, -a]])
        Ad = expm(A * Ts)
        Bd = np.linalg.solve(A, (Ad - np.eye(2))) / M.L
        ref = Ad @ z + Bd @ u
        got = propagate(z[0], z[1], u[0], u[1], w, Ts, M.R, M.L)
        assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_speed_loop_reproduces_thesis_droop():
    # Westin 2016 reports at most 450 rpm droop for a 1 Nm step; the speed
    # loop bandwidth in Setup is chosen to give about that
    s = Setup(fs=15e3)
    n = int(0.6 * s.fs)
    tb = np.zeros(n + 1)
    tb[int(0.1 * s.fs):] = 1.0
    w = simulate(param_vector(s), OBS_IDEAL, np.zeros(1), tb, np.zeros(n + 1), 0,
                 1, NO_DETECT, 0.0, False)[0]
    droop = 8500 - w.min() * 30 / math.pi
    assert 400 < droop < 500


@pytest.mark.parametrize("fs", [10e3, 15e3, 20e3])
@pytest.mark.parametrize("kind", [OBS_MRAS, OBS_EMF])
def test_observers_track_under_load_steps(fs, kind):
    s = Setup(fs=fs)
    m = s.motor
    op = (mras_params(s.Ts, m.R, m.L, m.psi, 400) if kind == OBS_MRAS
          else emf_params(s.Ts, m.R, m.L, m.psi, 1600, 400))
    n = int(0.5 * fs)
    tb = np.full(n + 1, 0.2)
    tb[int(0.1 * fs):int(0.3 * fs)] = 1.2
    w, wh, dth = simulate(param_vector(s), kind, op, tb, np.full(n + 1, 0.1), 1,
                          1, NO_DETECT, 0.0, False)[:3]
    assert np.degrees(np.sqrt(np.mean(dth ** 2))) < 1.0
    assert np.degrees(np.abs(dth).max()) < 5.0


def test_offline_detector_matches_in_loop_detector():
    rng = np.random.default_rng(3)
    Ts = 1 / 15e3
    w = 890 + np.cumsum(rng.normal(0, 0.5, 4000))
    w[2500:] -= np.arange(1500) * 3.0       # 45 krad/s^2 ramp
    for n in (1, 8, 30, 60):
        a_th = 30000.0
        k_off = detect_offline(w, n, Ts, a_th, 100.0)
        st = np.zeros(D_NMAX + 3)
        det_init(st, w[0])
        fired = [det_step(st, x, n, 1 / (n * Ts), a_th, 100.0) for x in w]
        k_in = fired.index(1) if 1 in fired else -1
        assert k_off == k_in


def _read_vector(path):
    hdr, rows = {}, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if "=" in line:
                k, v = line[2:].split("=", 1)
                hdr[k] = v
        else:
            rows.append([float(x) for x in line.split(",")])
    return hdr, np.array(rows)


def test_c_vectors_match_current_python_code():
    # The C tests compare against the outputs stored in c/vectors. Replay the
    # stored inputs through the current Python observer and detector to make
    # sure those outputs still are what the Python code produces. Re-running
    # the closed loop instead would not be portable: libm differs slightly
    # between platforms and the difference grows once the jammed rotor stalls.
    import export_vectors
    files = sorted((ROOT / "c" / "vectors").glob("*.csv"))
    assert len(files) == 4
    s = Setup(fs=export_vectors.FS)
    m = s.motor
    for f in files:
        hdr, rows = _read_vector(f)
        kind = OBS_MRAS if hdr["observer"] == "mras" else OBS_EMF
        bw = export_vectors.OBS_BW
        op = (mras_params(s.Ts, m.R, m.L, m.psi, bw) if kind == OBS_MRAS
              else emf_params(s.Ts, m.R, m.L, m.psi, 4 * bw, bw))
        stored_op = np.array([float(hdr[f"op{i}"]) for i in range(op.size)])
        assert np.allclose(op, stored_op, rtol=1e-12), f.name
        th, w = export_vectors.replay(kind, op, rows[:, :4], float(hdr["th0"]),
                                      float(hdr["w0"]), float(hdr["iq0"]), m.psi)
        dth = np.angle(np.exp(1j * (th - rows[:, 4])))
        assert np.abs(dth).max() < 1e-5, f.name
        assert np.abs(w - rows[:, 5]).max() < 1e-2, f.name
        n = int(hdr["det_n"])
        st = np.zeros(D_NMAX + 3)
        det_init(st, w[0] / m.p)
        fired = np.array([det_step(st, x / m.p, n, 1 / (n * s.Ts), float(hdr["det_ath"]),
                                   float(hdr["det_wen"])) for x in w])
        k_new = int(np.argmax(fired)) if fired.any() else -1
        k_old = int(np.argmax(rows[:, 6])) if rows[:, 6].any() else -1
        assert abs(k_new - k_old) <= 1, f.name
