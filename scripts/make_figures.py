"""Figures for the README and the note. Reads results/benchmark.json and
re-simulates the few runs it plots.

    python scripts/make_figures.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamsim.params import Setup, param_vector                                # noqa: E402
from jamsim.sim import simulate, OBS_IDEAL, OBS_MRAS, OBS_EMF, OBS_NAMES, NO_DETECT  # noqa: E402
from jamsim.benchmark import (Protocol, observer_params, run_normal, run_events,  # noqa: E402
                              peak_over, score_events, window_n, WINDOWS_MS, MARGIN,
                              FLOOR, summarise)
from jamsim.detector import window_decel                                     # noqa: E402
from jamsim.loads import event_run                                           # noqa: E402

FIG = ROOT / "results" / "figures"
COLORS = {OBS_IDEAL: "#6b6b6b", OBS_MRAS: "#2563eb", OBS_EMF: "#d97706"}
KINDS = (OBS_IDEAL, OBS_MRAS, OBS_EMF)
FS = 15e3

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6, "axes.edgecolor": "#888888", "legend.frameon": False,
    "lines.linewidth": 1.6,
})


def load_results():
    d = json.loads((ROOT / "results" / "benchmark.json").read_text())
    return {(r["fs"], r["observer"]): r for r in d["results"]}


def fig_severity(res):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, kind, xlabel in ((axes[0], "jam", "jam: final load torque [Nm]"),
                             (axes[1], "snag", "snag: added peak torque [Nm]")):
        for k in KINDS:
            r = res[(FS, k)]
            for label, ls in (("chosen", "-"), ("plain", "--")):
                summ = summarise(r[label]["events"])
                keys = sorted(x for x in summ if x[0] == kind)
                ax.plot([x[1] for x in keys], [100 * summ[x]["rate"] for x in keys], ls,
                        color=COLORS[k], marker="o", markersize=3.5,
                        label=f"{OBS_NAMES[k]}, {'tuned window' if label == 'chosen' else 'one sample'}")
        ax.set_xlabel(xlabel)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("detected [%]")
    axes[1].legend(loc="lower right", fontsize=7)
    fig.suptitle("Detection rate at 15 kHz (solid: tuned window, the three solid lines overlap at 100 %)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "detection_vs_severity.png")
    plt.close(fig)


def fig_window(res):
    """Evaluation-set detection rate and jam latency for every window."""
    s = Setup(fs=FS)
    proto = Protocol()
    w_floor = FLOOR * s.w_ref
    skip = int(0.05 * FS)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for k in KINDS:
        bw = res[(FS, k)]["chosen"]["bw"]
        op = observer_params(k, bw, s)
        set_a = run_normal(s, k, op, [10000 + i for i in range(proto.n_normal)], proto.normal_s)
        set_e = run_events(s, k, op, 500000, proto.n_eval)
        rate, lat = [], []
        windows = [ms for ms in WINDOWS_MS if ms > 0.0]
        for ms in windows:
            n = window_n(ms, FS)
            a_th = MARGIN * peak_over(set_a, n, s.Ts, w_floor, skip)
            rows = score_events(set_e, n, s.Ts, a_th, w_floor, FS, w_floor)
            jams = [r for r in rows if r[0] == "jam"]
            rate.append(100 * np.mean([r[2] for r in jams]))
            lat.append(np.median([r[3] for r in jams if r[2]]))
        x = windows
        axes[0].plot(x, rate, color=COLORS[k], marker="o", markersize=3.5, label=OBS_NAMES[k])
        axes[1].plot(x, lat, color=COLORS[k], marker="o", markersize=3.5, label=OBS_NAMES[k])
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_xlabel("detector window [ms]")
    axes[0].set_ylabel("jams detected [%]")
    axes[0].set_ylim(70, 102)
    axes[1].set_ylabel("median jam latency [ms]")
    axes[1].set_ylim(0, 3)
    axes[1].legend(fontsize=7)
    fig.suptitle("Window length against jam detection and latency (15 kHz, evaluation set, chosen bandwidth)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "window_tradeoff.png")
    plt.close(fig)


def fig_jam_example(res):
    s = Setup(fs=FS)
    P = param_vector(s)
    tb, tr, ev = event_run(np.random.default_rng(11), FS, "jam", 3.0, pre=0.30, post=0.03)
    t = (np.arange(tb.size - 1) - ev.k_on) / FS * 1e3
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True)
    rpm = 30 / math.pi
    true_w = None
    for k in KINDS:
        c = res[(FS, k)]["chosen"]
        op = observer_params(k, c["bw"], s)
        w, wh = simulate(P, k, op, tb, tr, 11, 1, NO_DETECT, 0.0, False)[:2]
        true_w = w if true_w is None else true_w
        a = window_decel(wh, c["n"], s.Ts) / 1e3
        k_fire = np.flatnonzero((a * 1e3 > c["a_th"]) & (wh > FLOOR * s.w_ref)
                                & (np.arange(a.size) > int(0.05 * FS)))
        if k != OBS_IDEAL:
            axes[0].plot(t, wh * rpm, color=COLORS[k], lw=1.1, label=f"{OBS_NAMES[k]} estimate")
        axes[1].plot(t, a, color=COLORS[k], lw=1.1,
                     label=f"{OBS_NAMES[k]}, {c['window_ms']:g} ms window")
        axes[1].axhline(c["a_th"] / 1e3, color=COLORS[k], lw=0.8, ls=":")
        if k_fire.size:
            for ax in axes:
                ax.axvline(t[k_fire[0]], color=COLORS[k], lw=0.8, ls="--")
    axes[0].plot(t, true_w * rpm, color="#111111", lw=1.0, ls="-", label="true speed", zorder=0)
    axes[0].set_ylabel("speed [rpm]")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[1].set_ylabel("window deceleration\n[krad/s$^2$]")
    axes[1].set_xlabel("time from start of torque rise [ms]")
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].set_xlim(-10, 25)
    axes[1].set_ylim(-20, 80)
    fig.suptitle(f"A 3 Nm jam rising in {ev.shape_ms:.1f} ms from a {ev.level:.2f} Nm cut; dashed lines mark the detector firing, dotted lines the thresholds", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "jam_example.png")
    plt.close(fig)


def fig_brake(res):
    s = Setup(fs=FS)
    P = param_vector(s)
    k = OBS_MRAS
    c = res[(FS, k)]["chosen"]
    op = observer_params(k, c["bw"], s)
    tb, tr, ev = event_run(np.random.default_rng(23), FS, "snag", 3.0, pre=0.30, post=0.12)
    t = (np.arange(tb.size - 1) - ev.k_on) / FS * 1e3
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for brake, color, label in ((False, "#9ca3af", "no detection: drive keeps pushing"),
                                (True, COLORS[k], "MRAS + detector, then brake at 60 A")):
        w, wh, dth, iq, iqr, fk, _ = simulate(P, k, op, tb, tr, 23, c["n"],
                                              c["a_th"] if brake else NO_DETECT,
                                              FLOOR * s.w_ref, brake)
        revs = np.sum(np.maximum(w[ev.k_on:], 0)) / FS / (2 * math.pi)
        ax.plot(t, w * 30 / math.pi, color=color, label=f"{label} ({revs:.1f} rev in {t[-1]:.0f} ms)")
        if brake and fk >= 0:
            ax.axvline(t[fk], color=color, lw=0.8, ls="--")
    ax.axvspan(0, ev.shape_ms, color="#fde68a", alpha=0.5, lw=0, label=f"{ev.severity:g} Nm snag, {ev.shape_ms:.1f} ms")
    ax.set_xlim(-10, t[-1])
    ax.set_xlabel("time from start of snag [ms]")
    ax.set_ylabel("rotor speed [rpm]")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "brake_example.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = load_results()
    fig_severity(res)
    fig_jam_example(res)
    fig_brake(res)
    fig_window(res)
    print("figures in", FIG.relative_to(ROOT))


if __name__ == "__main__":
    main()
