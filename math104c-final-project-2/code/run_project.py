"""
run_project.py
Generates every figure and LaTeX table used in the report for Final Project 2.

Outputs:
    ../figures/*.png
    ../tables/*.tex   (table bodies, \\input-ed by the report)

Also prints a summary block of numbers that are quoted in the prose.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pde_methods as m

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")

plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


def write_table(fname, body):
    with open(os.path.join(TAB, fname), "w") as fh:
        fh.write(body)
    print("wrote", fname)


# ----------------------------------------------------------------------
# Problem A: Laplace on the unit square
# ----------------------------------------------------------------------
def problem_A():
    g_bottom = lambda x: np.zeros_like(x)          # u(x, 0) = 0
    g_left = lambda y: np.zeros_like(y)            # u(0, y) = 0
    g_top = lambda x: 100.0 * x                    # u(x, 1) = 100 x
    g_right = lambda y: 100.0 * y                  # u(1, y) = 100 y

    # The exact analytic solution of this BVP is u(x,y) = 100 x y. It is
    # harmonic and matches all four boundary conditions, including the corner
    # (1,1) where the two non-zero edges agree on 100. The Fourier series
    # version equals 100xy in the interior up to roundoff but suffers from
    # Gibbs ringing on the y=1 / x=1 edges, so we use the closed form 100xy
    # as the reference for error tables.
    hs = [1 / 4, 1 / 8, 1 / 16]
    results = {}
    for h in hs:
        x, y, W, it = m.laplace_5pt_gs(h, g_bottom, g_top, g_left, g_right,
                                       tol=1e-8, maxit=50000)
        U = 100.0 * np.outer(x, y)
        results[h] = dict(x=x, y=y, W=W, U=U, it=it)
        print(f"  Laplace h={h:.4f}: Gauss-Seidel converged in {it} iters,"
              f" max error = {np.max(np.abs(W[1:-1, 1:-1] - U[1:-1, 1:-1])):.3e}")

    # ------------------------------------------------------------------
    # Table A1: at h = 1/4, list the four interior nodes
    #   exact / approx / |error|
    # ------------------------------------------------------------------
    r = results[1 / 4]
    x, y, W, U = r["x"], r["y"], r["W"], r["U"]
    lines = []
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            lines.append(
                f"({x[i]:.2f}, {y[j]:.2f}) & "
                f"{U[i, j]:.4f} & {W[i, j]:.4f} & {abs(W[i,j]-U[i,j]):.2e} \\\\"
            )
    body = (r"\begin{tabular}{l r r r}\hline" + "\n"
            + r"$(x_i, y_j)$ & exact $u$ & approx $w_{ij}$ & $|u - w_{ij}|$ \\"
            + "\n\\hline\n" + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")
    write_table("tabA_h4.tex", body)

    # ------------------------------------------------------------------
    # Table A2: refinement study
    #   h, max error, observed order, Gauss-Seidel iterations
    # ------------------------------------------------------------------
    prev_err = None
    lines = []
    for h in hs:
        r = results[h]
        err = np.max(np.abs(r["W"][1:-1, 1:-1] - r["U"][1:-1, 1:-1]))
        if prev_err is None:
            order_str = "--"
        else:
            order_str = f"{np.log2(prev_err / err):.2f}"
        lines.append(
            f"$1/{int(round(1/h))}$ & {err:.3e} & {order_str} & {r['it']} \\\\"
        )
        prev_err = err
    body = (r"\begin{tabular}{r r r r}\hline" + "\n"
            + r"$h$ & max error & observed order & GS iterations \\"
            + "\n\\hline\n" + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")
    write_table("tabA_refine.tex", body)

    # ------------------------------------------------------------------
    # Figure A: contour of the exact 100xy (left) and absolute error at
    # h=1/16 (right). Error is ~1e-7, set by the iteration tolerance.
    # ------------------------------------------------------------------
    r = results[1 / 16]
    X, Y = np.meshgrid(r["x"], r["y"], indexing="ij")
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.0))
    cs = ax[0].contourf(X, Y, r["U"], levels=20, cmap="viridis")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")
    ax[0].set_title("Problem A: exact $u(x,y) = 100\\,x\\,y$")
    fig.colorbar(cs, ax=ax[0])
    err = np.abs(r["W"] - r["U"])
    cs2 = ax[1].contourf(X, Y, err, levels=20, cmap="magma")
    ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$y$")
    ax[1].set_title("Problem A: $|u - w_{ij}|$ at $h=1/16$")
    fig.colorbar(cs2, ax=ax[1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA.png")); plt.close(fig)
    print("wrote figA.png")

    # ------------------------------------------------------------------
    # Bonus comparison: Gauss-Seidel vs optimal SOR iteration counts
    # (referenced in the discussion)
    # ------------------------------------------------------------------
    print("  --- iteration count: Gauss-Seidel vs optimal SOR ---")
    for h in hs:
        _, _, _, it_sor, w = m.laplace_5pt_sor(
            h, g_bottom, g_top, g_left, g_right, tol=1e-8, maxit=50000)
        print(f"    h=1/{int(round(1/h))}:  GS={results[h]['it']:5d}  "
              f"SOR(omega={w:.3f})={it_sor:5d}")


# ----------------------------------------------------------------------
# Problem B: heat equation
# ----------------------------------------------------------------------
def problem_B():
    u0 = lambda x: np.sin(np.pi * x)
    exact = lambda x, t: np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x)

    T = 0.5
    h = 0.1
    cases = [
        ("Forward (stable)",   "fwd_s", m.heat_forward,        h, 0.0025),  # lam=0.25
        ("Forward (unstable)", "fwd_u", m.heat_forward,        h, 0.01),    # lam=1.0  > 0.5
        ("Backward",           "bwd",   m.heat_backward,       h, 0.01),
        ("Crank-Nicolson",     "cn",    m.heat_crank_nicolson, h, 0.01),
    ]

    runs = {}
    for label, tag, solver, hh, kk in cases:
        x, t, W, lam = solver(hh, kk, T, u0)
        runs[tag] = dict(label=label, x=x, t=t, W=W, lam=lam, h=hh, k=kk)
        Ue = np.outer(np.sin(np.pi * x), np.exp(-np.pi ** 2 * t))
        err = np.max(np.abs(W - Ue))
        print(f"  {label:22s}  h={hh}, k={kk}, lam={lam:.3f}, "
              f"max|err| over [0,T]={err:.3e}, w(0.5,T)={W[5,-1]:.6e}")

    # ------------------------------------------------------------------
    # Table B1: at t = 0.5, x = 0.1..0.9, exact + each method + |error|
    # for the three "good" runs (forward-stable, backward, CN).
    # ------------------------------------------------------------------
    x_fwd = runs["fwd_s"]["x"]
    interior = list(range(1, len(x_fwd) - 1))     # 1..9
    ue = np.sin(np.pi * x_fwd) * np.exp(-np.pi ** 2 * T)
    lines = []
    for i in interior:
        row = [f"{x_fwd[i]:.1f}", f"{ue[i]:.6f}"]
        for tag in ("fwd_s", "bwd", "cn"):
            wi = runs[tag]["W"][i, -1]
            row += [f"{wi:.6f}", f"{abs(wi-ue[i]):.2e}"]
        lines.append(" & ".join(row) + r" \\")
    header = (r"$x_i$ & exact & FD-stab & $|e|$ & BD & $|e|$ & CN & $|e|$ \\")
    body = (r"\begin{tabular}{r r r r r r r r}\hline" + "\n"
            + header + "\n\\hline\n" + "\n".join(lines)
            + "\n\\hline\n\\end{tabular}\n")
    write_table("tabB_values.tex", body)

    # ------------------------------------------------------------------
    # Table B2: max |error| over all (x, t) for each scheme
    # ------------------------------------------------------------------
    lines = []
    for tag in ("fwd_s", "fwd_u", "bwd", "cn"):
        r = runs[tag]
        Ue = np.outer(np.sin(np.pi * r["x"]),
                      np.exp(-np.pi ** 2 * r["t"]))
        err = np.max(np.abs(r["W"] - Ue))
        if not np.isfinite(err) or err > 1e6:
            err_str = "blow-up"
        else:
            err_str = f"{err:.3e}"
        lines.append(f"{r['label']} & {r['h']} & {r['k']} & "
                     f"{r['lam']:.3f} & {err_str} \\\\")
    body = (r"\begin{tabular}{l r r r r}\hline" + "\n"
            + r"Method & $h$ & $k$ & $\lambda$ & max $|u - w|$ \\"
            + "\n\\hline\n" + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")
    write_table("tabB_errors.tex", body)

    # ------------------------------------------------------------------
    # Figure B: three panels
    #   (1) profile at t=0.5, exact vs FD-stable, BD, CN (all on top of each other)
    #   (2) FD-unstable: profile at a few times, showing oscillation/blow-up
    #   (3) error vs time, log scale, for the three good schemes
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    xs = runs["fwd_s"]["x"]
    xf = np.linspace(0, 1, 200)
    ax[0].plot(xf, exact(xf, T), "k-", lw=2, label="exact")
    for tag, mk in [("fwd_s", "o"), ("bwd", "s"), ("cn", "d")]:
        ax[0].plot(xs, runs[tag]["W"][:, -1], mk, ms=4,
                   label=runs[tag]["label"])
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u(x, T)$")
    ax[0].set_title(f"Problem B: profile at $t = {T}$")
    ax[0].legend(fontsize=8)

    # FD unstable: plot at successive times to show oscillation
    r = runs["fwd_u"]
    ax[1].plot(xf, exact(xf, 0.0), "k-", lw=1, label="exact $t=0$")
    for jt, ls in [(0, "k-"), (5, "C0o-"), (20, "C1s-"), (40, "C3^-")]:
        if jt < r["W"].shape[1]:
            ax[1].plot(r["x"], r["W"][:, jt], ls, ms=4,
                       label=f"$t = {r['t'][jt]:.3f}$")
    ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$w$")
    ax[1].set_title(f"FD unstable, $\\lambda = {r['lam']:.2f}$")
    ax[1].legend(fontsize=8)

    # error vs time, log scale (skip t=0 where error is 0 by construction)
    for tag, mk in [("fwd_s", "o-"), ("bwd", "s-"), ("cn", "d-")]:
        r = runs[tag]
        Ue = np.outer(np.sin(np.pi * r["x"]),
                      np.exp(-np.pi ** 2 * r["t"]))
        err_t = np.max(np.abs(r["W"] - Ue), axis=0)
        ax[2].semilogy(r["t"][1:], err_t[1:], mk, ms=3, label=r["label"])
    ax[2].set_xlabel("$t$"); ax[2].set_ylabel("max$_x |u - w|$")
    ax[2].set_title("Problem B: error vs $t$")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figB.png")); plt.close(fig)
    print("wrote figB.png")


# ----------------------------------------------------------------------
# Problem C1: multi-mode initial condition
# ----------------------------------------------------------------------
def problem_C1():
    """Heat equation with u(x,0) = sin(pi x) + 0.5 sin(3 pi x).

    Exact: u(x,t) = e^{-pi^2 t} sin(pi x) + 0.5 e^{-9 pi^2 t} sin(3 pi x).
    The k=9 mode decays nine times faster than the k=1 mode, so even though it
    starts with half the amplitude, it is essentially gone by t = 0.05.
    """
    u0 = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    exact = lambda x, t: (np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x)
                          + 0.5 * np.exp(-9 * np.pi ** 2 * t) * np.sin(3 * np.pi * x))

    T = 0.2
    h, k = 0.05, 0.005          # CN is unconditionally stable
    x, t, W, lam = m.heat_crank_nicolson(h, k, T, u0)
    print(f"  C1 (CN): h={h}, k={k}, lam={lam:.3f}, "
          f"max|err|={np.max(np.abs(W - np.array([exact(xi, t) for xi in x]))):.3e}")

    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8))
    xf = np.linspace(0, 1, 300)
    times = [0.0, 0.005, 0.02, 0.05, 0.2]
    cmap = plt.get_cmap("viridis")
    for kk, tt in enumerate(times):
        col = cmap(kk / (len(times) - 1))
        ax[0].plot(xf, exact(xf, tt), "-", color=col, lw=2,
                   label=f"exact $t={tt}$")
        # numerical profile at the closest available time
        jt = int(round(tt / k))
        ax[0].plot(x, W[:, jt], "o", color=col, ms=4)
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u(x, t)$")
    ax[0].set_title("Problem C1: multi-mode IC, CN ($h=0.05$, $k=0.005$)")
    ax[0].legend(fontsize=8)

    # plot Fourier amplitudes |a_n(t)| extracted by discrete sine transform
    # (just modes n=1 and n=3 for clarity)
    m_int = len(x) - 2
    sn1 = np.sin(np.pi * x[1:-1])
    sn3 = np.sin(3 * np.pi * x[1:-1])
    a1_num = 2 * (W[1:-1, :].T @ sn1) / (m_int + 1) * (m_int + 1) / (m_int + 1)
    # simpler: trapezoidal integral approximation of (2/L) int u sin(n pi x) dx
    a1_num = 2 * np.trapz(W * np.sin(np.pi * x)[:, None], x=x, axis=0)
    a3_num = 2 * np.trapz(W * np.sin(3 * np.pi * x)[:, None], x=x, axis=0)
    a1_ex = np.exp(-np.pi ** 2 * t)
    a3_ex = 0.5 * np.exp(-9 * np.pi ** 2 * t)
    ax[1].semilogy(t, np.abs(a1_ex), "k-", lw=2, label="exact $|a_1|$")
    ax[1].semilogy(t, np.abs(a1_num), "C0o", ms=3, label="CN $|a_1|$")
    ax[1].semilogy(t, np.abs(a3_ex), "k--", lw=2, label="exact $|a_3|$")
    ax[1].semilogy(t, np.abs(a3_num), "C3s", ms=3, label="CN $|a_3|$")
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel("mode amplitude")
    ax[1].set_title("Problem C1: high modes decay faster")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figC1.png")); plt.close(fig)
    print("wrote figC1.png")

    # Table: at t = 0.05, compare exact vs CN profile
    t_eval = 0.05
    jt = int(round(t_eval / k))
    lines = []
    interior = list(range(1, len(x) - 1))
    pick = interior[::3]      # every third interior point for compactness
    for i in pick:
        ue = exact(x[i], t_eval)
        wi = W[i, jt]
        lines.append(f"{x[i]:.2f} & {ue:.6f} & {wi:.6f} & {abs(wi-ue):.2e} \\\\")
    body = (r"\begin{tabular}{r r r r}\hline" + "\n"
            + r"$x_i$ & exact $u(x_i, 0.05)$ & CN $w_i$ & $|u - w_i|$ \\"
            + "\n\\hline\n" + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")
    write_table("tabC1.tex", body)


# ----------------------------------------------------------------------
# Problem C2: Poisson equation with a manufactured solution
# ----------------------------------------------------------------------
def problem_C2():
    """u_xx + u_yy = f(x,y) on (0,1)^2 with u = 0 on the boundary,
    where the right-hand side is chosen so that u(x,y) = sin(pi x) sin(pi y).
    Plugging in, f(x,y) = -2 pi^2 sin(pi x) sin(pi y).

    Goal: confirm the 5-point stencil is O(h^2).
    """
    u_ex = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    f_rhs = lambda x, y: -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    zero = lambda v: np.zeros_like(v)

    hs = [1 / 4, 1 / 8, 1 / 16, 1 / 32]
    rows = []
    errs = []
    iters = []
    prev = None
    for h in hs:
        x, y, W, it = m.poisson_5pt_gs(h, f_rhs, zero, zero, zero, zero,
                                       tol=1e-9, maxit=80000)
        U = np.outer(np.sin(np.pi * x), np.sin(np.pi * y))
        err = np.max(np.abs(W[1:-1, 1:-1] - U[1:-1, 1:-1]))
        if prev is None:
            order_str = "--"
        else:
            order_str = f"{np.log2(prev / err):.2f}"
        rows.append(
            f"$1/{int(round(1/h))}$ & {err:.3e} & {order_str} & {it} \\\\"
        )
        errs.append(err); iters.append(it); prev = err
        print(f"  C2 Poisson h={h:.4f}: GS iters={it}, max|err|={err:.3e}")

    body = (r"\begin{tabular}{r r r r}\hline" + "\n"
            + r"$h$ & max $|u - w|$ & observed order & GS iterations \\"
            + "\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    write_table("tabC2.tex", body)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    ax.loglog(hs, errs, "o-", lw=2, label="max $|u - w|$")
    ax.loglog(hs, [errs[0] * (h / hs[0]) ** 2 for h in hs], "k--",
              label="$O(h^2)$ reference")
    ax.set_xlabel("$h$"); ax.set_ylabel("max error")
    ax.set_title("Problem C2: 5-point stencil convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figC2.png")); plt.close(fig)
    print("wrote figC2.png")


if __name__ == "__main__":
    print("=== Problem A: Laplace ===")
    problem_A()
    print("\n=== Problem B: heat equation ===")
    problem_B()
    print("\n=== Problem C1: multi-mode IC ===")
    problem_C1()
    print("\n=== Problem C2: Poisson, convergence ===")
    problem_C2()
    print("\ndone.")
