"""Step 2: forecast a disturbance and compare restoration actions (Figure 5B-D).

Network-metacommunity gLV model for species i in habitat p:

    dN_ip/dt = N_ip (r_i + sum_j A_ij N_jp - N_ip / K_pi)
               + sum_q D_i,pq N_iq - N_ip sum_q D_i,qp

K_pi comes from habitats.csv (summed SDM suitability; 1 where the species is
absent), D_i = M0 * dispersal_<species>.csv with a zero diagonal. The model is
integrated over unit time steps; between steps K is perturbed by AR(1)
environmental noise and abundances by lognormal demographic noise.

Scenario: the H2-H9 corridor is lost at t = 5.
  Figure 5B  Species A in H9 with vs without the corridor (t = 0..19).
  Figure 5C/D  restoration at t = 20 (t = 0..40):
     A0  no restoration
     A1  rebuild the H2-H9 corridor
     A2a reintroduce Species A in H9 only
     A2b improve habitat quality in H9 (raise K of Species A) + reintroduce;
         K is calibrated so that A2b sustains the same level as A1.

Run:  python restoration_forecast.py                      # tables used in the paper (data/network)
      python restoration_forecast.py outputs/network      # tables from habitat_network.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.abspath(__file__))
NET_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "network")
OUT_DIR = os.path.join(ROOT, "outputs")

# ---- parameters ----------------------------------------------------------- #
SPECIES = ["Bilberry", "Heather", "Western_gorse"]      # shown as Species A, B, C
LABEL = ["Species A", "Species B", "Species C"]
R = np.array([0.8, 0.9, 0.7])                          # intrinsic growth rates
COMP = np.array([[0.00, 0.15, 0.10],                   # competition of j on i,
                 [0.15, 0.00, 0.22],                   # relative to the mean K
                 [0.10, 0.22, 0.00]])
M0 = 0.10            # emigration rate scaling the dispersal kernel
K_FLOOR = 1.0        # K where a species is absent (a sink)
SIGMA_ENV, PHI_ENV = 0.15, 0.70   # AR(1) environmental noise on K
SIGMA_DEMO = 0.05                 # lognormal demographic noise per step
SEED = 2025

FOCAL = 0                    # Species A
H_SOURCE, H_SINK = 2, 9      # corridor H2-H9
T_LOSS, T_RESTORE = 5, 20

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 18,
                     "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16})


# ---- model ---------------------------------------------------------------- #
def load_network():
    hab = pd.read_csv(os.path.join(NET_DIR, "habitats.csv")).sort_values("habitat_id")
    K = hab[[f"K_{s}" for s in SPECIES]].to_numpy(float)
    K[K <= 0] = K_FLOOR
    D = []
    for s in SPECIES:
        d = M0 * pd.read_csv(os.path.join(NET_DIR, f"dispersal_{s}.csv"), index_col=0).to_numpy(float)
        np.fill_diagonal(d, 0.0)
        D.append(d)
    A = -COMP * R[:, None] / K[K > K_FLOOR].mean()
    return K, np.stack(D), A, hab["habitat_id"].astype(int).tolist()


def step(N, K, D, A, t0, t1):
    """Integrate one time step (LSODA) and return abundances at t1."""
    emig = D.sum(axis=1)                                    # (S, P)

    def rhs(t, y):
        N = np.maximum(y.reshape(K.shape), 0.0)             # (P, S)
        local = N * (R + N @ A.T - N / K)
        immig = np.einsum("spq,qs->ps", D, N)
        return (local + immig - N * emig.T).ravel()

    sol = solve_ivp(rhs, (t0, t1), N.ravel(), method="LSODA", t_eval=[t1], rtol=1e-6, atol=1e-9)
    N = sol.y[:, -1].reshape(K.shape)
    N = np.maximum(np.where(np.isfinite(N), N, 0.0), 0.0)
    N[N < 1e-10] = 0.0
    return N


def simulate(K, n_steps, corridor_open, D_full, D_cut, A, demo_rng, rng_env,
             K_after=None, pulse=0.0, i_sink=None):
    """Shared environmental noise from rng_env; returns trajectory (T, P, S)."""
    P, S = K.shape
    z = np.zeros((n_steps, P, S))
    z[0] = rng_env.standard_normal((P, S))
    e = rng_env.standard_normal((n_steps, P, S))
    for w in range(1, n_steps):
        z[w] = PHI_ENV * z[w - 1] + np.sqrt(1 - PHI_ENV ** 2) * e[w]
    env = np.exp(SIGMA_ENV * z - SIGMA_ENV ** 2 / 2)

    N = np.where(K > K_FLOOR, rng_env.uniform(0.2, 0.6, (P, S)) * K, 0.0)
    demo_rng = demo_rng or rng_env
    traj = [N.copy()]
    for w in range(n_steps):
        if pulse and w == T_RESTORE:
            N[i_sink, FOCAL] += pulse
        D = D_full if corridor_open(w) else D_cut
        Kw = K_after if (K_after is not None and w >= T_RESTORE) else K
        N = step(N, Kw * env[w], D, A, w, w + 1)
        N = np.where(N > 0, N * np.exp(SIGMA_DEMO * demo_rng.standard_normal(N.shape)), N)
        traj.append(N.copy())
    return np.array(traj)


# ---- plotting ------------------------------------------------------------- #
def smooth(ax, t, y, **kw):
    """PCHIP-smoothed line; a repeated time point is drawn as a vertical jump."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    cuts = np.r_[0, np.where(np.diff(t) <= 0)[0] + 1, len(t)]
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        tt = np.linspace(t[a], t[b - 1], 300 if len(cuts) == 2 else 150)
        ax.plot(tt, PchipInterpolator(t[a:b], y[a:b])(tt), **(kw if k == 0 else {**kw, "label": None}))
        if b < len(t):
            ax.plot(t[b - 1:b + 1], y[b - 1:b + 1], **{**kw, "label": None})


def fig5B(t, with_c, without_c):
    fig, ax = plt.subplots(figsize=(6, 6))
    ts = np.linspace(t[0], t[-1], 300)
    ax.fill_between(ts, PchipInterpolator(t, without_c)(ts), PchipInterpolator(t, with_c)(ts),
                    color="#1f77b4", alpha=0.2, label="Abundance lost")
    smooth(ax, t, with_c, color="#1f77b4", lw=2.2, label="with H2–H9 corridor")
    smooth(ax, t, without_c, color="#d62728", lw=2.2, label="H2–H9 corridor lost")
    ymax = 1.15 * max(with_c.max(), without_c.max())
    ax.axvline(T_LOSS, color="0.4", ls=":", lw=2)
    ax.text(T_LOSS + 0.3, 0.98 * ymax, f"corridor lost (t={T_LOSS})", color="0.3", fontsize=14, va="top")
    loss = (without_c[-1] - with_c[-1]) / with_c[-1]
    ax.annotate("", xy=(t[-1], without_c[-1]), xytext=(t[-1], with_c[-1]),
                arrowprops=dict(arrowstyle="<->", lw=1.8))
    ax.text(t[-1] - 0.4, (with_c[-1] + without_c[-1]) / 2, f"{loss:+.0%}", ha="right", va="center",
            fontsize=20, fontweight="bold")
    ax.set(xlabel="time", ylabel=f"{LABEL[FOCAL]} abundance", ylim=(0, ymax))
    ax.legend(loc="lower right", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5B_disturbance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return loss


def fig5CD(t, a0, a1, a2a, a2b, pulse, K_star):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.0), sharey=True)
    ymax = 1.18 * max(a0.max(), a1.max(), a2a.max(), a2b.max(), a2b[T_RESTORE] + pulse)
    ts = np.linspace(t[0], t[-1], 150)
    # A2b is drawn with the reintroduction pulse as a vertical jump at t = 20
    t_b = np.r_[t[:T_RESTORE + 1], t[T_RESTORE], t[T_RESTORE + 1:]]
    y_b = np.r_[a2b[:T_RESTORE + 1], a2b[T_RESTORE] + pulse, a2b[T_RESTORE + 1:]]
    lines = [("A2a: Reintroduce {:.0f} of A only (no habitat improvement)".format(pulse), t, a2a, "#1f77b4"),
             ("A2b: Improve K(H9)→{:.1f} (+reintroduce {:.0f} of A)".format(K_star, pulse), t_b, y_b, "#ff7f0e")]
    for ax, (lab, tx, y, col) in zip(axes, lines):
        ax.fill_between(ts, PchipInterpolator(t, a0)(ts), PchipInterpolator(t, a1)(ts), color="#2ca02c",
                        alpha=0.12, label="Difference between rebuilding corridor and no restoration")
        smooth(ax, t, a0, ls="--", lw=2, color="black", label="A0: No restoration")
        smooth(ax, t, a1, lw=2.4, color="#2ca02c", alpha=0.5, label="A1: Rebuild H2–H9 corridor")
        smooth(ax, tx, y, lw=2.4, color=col, alpha=0.7, label=lab)
        for x, txt, ls in [(T_LOSS, f"corridor lost\n(t={T_LOSS})", ":"),
                           (T_RESTORE, f"restoration\n(t={T_RESTORE})", "-.")]:
            ax.axvline(x, color="0.4", ls=ls, lw=2, zorder=0)
            ax.text(x + 0.3, 0.02 * ymax, txt, color="0.3", fontsize=11, va="bottom")
        ax.set(xlim=(t[0], t[-1]), ylim=(0, ymax))
    axes[0].set_ylabel(f"{LABEL[FOCAL]} abundance in H{H_SINK}")
    h = {l: hd for ax in axes for hd, l in zip(*ax.get_legend_handles_labels())}
    order = ["A0: No restoration", "A1: Rebuild H2–H9 corridor", lines[0][0], lines[1][0],
             "Difference between rebuilding corridor and no restoration"]
    axes[0].legend([h[l] for l in order], order, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=13)
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(os.path.join(OUT_DIR, "fig5CD_restoration.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- main ----------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    K, D_full, A, hab_ids = load_network()
    i_src, i_sink = hab_ids.index(H_SOURCE), hab_ids.index(H_SINK)
    D_cut = D_full.copy()
    D_cut[:, i_src, i_sink] = D_cut[:, i_sink, i_src] = 0.0

    # Figure 5B: corridor lost at t = 5 vs corridor kept (same noise realisation)
    def run_B(open_rule):
        return simulate(K, 19, open_rule, D_full, D_cut, A, None, np.random.default_rng(SEED))
    kept = run_B(lambda w: True)[:, i_sink, FOCAL]
    lost = run_B(lambda w: w < T_LOSS)[:, i_sink, FOCAL]
    loss = fig5B(np.arange(20.0), kept, lost)
    print(f"5B: {LABEL[FOCAL]} in H{H_SINK} at t=19: {kept[-1]:.2f} -> {lost[-1]:.2f} ({loss:+.1%})")

    # Figures 5C-D: restoration at t = 20
    def run_CD(scenario, K_after=None, pulse=0.0):
        rules = {"no_restore": lambda w: w < T_LOSS,
                 "corridor": lambda w: not (T_LOSS <= w < T_RESTORE),
                 "reintro": lambda w: w < T_LOSS}
        demo = np.random.default_rng(SEED + {"no_restore": 1002, "corridor": 1003, "reintro": 1004}[scenario])
        return simulate(K, 40, rules[scenario], D_full, D_cut, A, demo, np.random.default_rng(SEED),
                        K_after=K_after, pulse=pulse, i_sink=i_sink)[:, i_sink, FOCAL]

    def improved_K(k):
        Kk = K.copy()
        Kk[i_sink, FOCAL] = k
        return Kk

    def sustained(x):                  # mean abundance over t >= 25
        return x[T_RESTORE + 5:].mean()

    a0, a1 = run_CD("no_restore"), run_CD("corridor")
    target = sustained(a1)
    lo, hi = K_FLOOR, 12.5             # bisection: K of Species A in H9 that matches A1
    for _ in range(34):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sustained(run_CD("reintro", improved_K(mid))) < target else (lo, mid)
    K_star = (lo + hi) / 2
    pulse = max(target - a0[T_RESTORE], 0.0)
    a2a = run_CD("reintro", K, pulse)
    a2b = run_CD("reintro", improved_K(K_star), pulse)
    fig5CD(np.arange(41.0), a0, a1, a2a, a2b, pulse, K_star)
    print(f"5C-D: sustained level A0 {sustained(a0):.2f}, A1 {target:.2f}; "
          f"reintroduce {pulse:.1f}; equivalent K of {LABEL[FOCAL]} in H{H_SINK} = {K_star:.2f}")
    print(f"Figures saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
