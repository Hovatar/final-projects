"""
run_project.py
Generates every figure and LaTeX table used in the report for Final Project 1.

Outputs:
    ../figures/*.png
    ../tables/*.tex   (table bodies, \input-ed by the report)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ode_methods as m

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")

plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


# ----------------------------------------------------------------------
# Problem definitions
# ----------------------------------------------------------------------
# A:  y' = y - t^2 + 1,  y(0)=0.5
fA = lambda t, y: y - t**2 + 1
fAprime = lambda t, y: (y - t**2 + 1) - 2 * t          # f' = f_t + f_y f
yA = lambda t: (t + 1)**2 - 0.5 * np.exp(t)

# B:  y' = 2y,  y(0)=1
fB = lambda t, y: 2 * y
fBprime = lambda t, y: 4 * y
yB = lambda t: np.exp(2 * t)

# C (student-designed):  y' = -10y + sin t,  y(0)=1   (stiff-ish, rapid decay)
fC = lambda t, y: -10 * y + np.sin(t)
fCprime = lambda t, y: np.cos(t) - 10 * (-10 * y + np.sin(t))
yC = lambda t: (10 * np.sin(t) - np.cos(t)) / 101 + (102 / 101) * np.exp(-10 * t)


def solve_all(f, fprime, a, b, alpha, N):
    """Return {name: (t, w)} for all eight methods."""
    out = {}
    out["Euler"] = m.euler(f, a, b, alpha, N)
    out["Taylor 2"] = m.taylor2(f, fprime, a, b, alpha, N)
    out["Midpoint"] = m.midpoint(f, a, b, alpha, N)
    out["Modified Euler"] = m.modified_euler(f, a, b, alpha, N)
    out["Heun (order 3)"] = m.heun3(f, a, b, alpha, N)
    out["RK4"] = m.rk4(f, a, b, alpha, N)
    out["Adams-Bashforth 4"] = m.adams_bashforth4(f, a, b, alpha, N)
    out["Adams PC 4"] = m.adams_pc4(f, a, b, alpha, N)
    return out


def write_table(fname, body):
    with open(os.path.join(TAB, fname), "w") as fh:
        fh.write(body)
    print("wrote", fname)


# ----------------------------------------------------------------------
# Problem A
# ----------------------------------------------------------------------
def problem_A():
    a, b, alpha, N = 0.0, 2.0, 0.5, 10           # h = 0.2
    sols = solve_all(fA, fAprime, a, b, alpha, N)
    t = sols["Euler"][0]
    exact = yA(t)

    cols = ["Euler", "Midpoint", "Heun (order 3)", "RK4", "Adams PC 4"]
    short = {"Euler": "Euler", "Midpoint": "Midpoint",
             "Heun (order 3)": "Heun", "RK4": "RK4", "Adams PC 4": "Adams PC"}

    # Error table at every grid point.
    lines = []
    for i in range(N + 1):
        row = [f"{t[i]:.1f}", f"{exact[i]:.7f}"]
        for c in cols:
            row.append(f"{abs(sols[c][1][i] - exact[i]):.2e}")
        lines.append(" & ".join(row) + r" \\")
    header = "$t_i$ & $y(t_i)$ & " + " & ".join(short[c] for c in cols) + r" \\"
    body = (r"\begin{tabular}{r r " + "r" * len(cols) + "}\n\\hline\n"
            + header + "\n\\hline\n" + "\n".join(lines)
            + "\n\\hline\n\\end{tabular}\n")
    write_table("tabA_errors.tex", body)

    # Convergence study: max error vs h, with observed order.
    Ns = [10, 20, 40, 80, 160]
    methods = ["Euler", "Midpoint", "Heun (order 3)", "RK4", "Adams PC 4"]
    maxerr = {name: [] for name in methods}
    for Nk in Ns:
        s = solve_all(fA, fAprime, a, b, alpha, Nk)
        tk = s["Euler"][0]
        ek = yA(tk)
        for name in methods:
            maxerr[name].append(np.max(np.abs(s[name][1] - ek)))

    lines = []
    for k, Nk in enumerate(Ns):
        h = (b - a) / Nk
        row = [f"{h:.4f}"]
        for name in methods:
            e = maxerr[name][k]
            if k == 0:
                row.append(f"{e:.2e} (--)")
            else:
                p = np.log2(maxerr[name][k - 1] / e)
                row.append(f"{e:.2e} ({p:.2f})")
        lines.append(" & ".join(row) + r" \\")
    header = "$h$ & " + " & ".join(short[c] for c in methods) + r" \\"
    body = (r"\begin{tabular}{r " + "r" * len(methods) + "}\n\\hline\n"
            + header + "\n\\hline\n" + "\n".join(lines)
            + "\n\\hline\n\\end{tabular}\n")
    write_table("tabA_order.tex", body)

    # Compact "all eight methods" table: max error at h=0.1 with observed order
    # (order estimated from the h=0.2 -> h=0.1 refinement).
    all_methods = ["Euler", "Taylor 2", "Midpoint", "Modified Euler",
                   "Heun (order 3)", "RK4", "Adams-Bashforth 4", "Adams PC 4"]
    label = {"Euler": "Euler", "Taylor 2": "Taylor (order 2)",
             "Midpoint": "Midpoint", "Modified Euler": "Modified Euler",
             "Heun (order 3)": "Heun (order 3)", "RK4": "RK4",
             "Adams-Bashforth 4": "Adams--Bashforth 4", "Adams PC 4": "Adams PC 4"}
    s10 = solve_all(fA, fAprime, a, b, alpha, 10)   # h = 0.2
    s20 = solve_all(fA, fAprime, a, b, alpha, 20)   # h = 0.1
    e10 = yA(s10["Euler"][0]); e20 = yA(s20["Euler"][0])
    lines = []
    for name in all_methods:
        m10 = np.max(np.abs(s10[name][1] - e10))
        m20 = np.max(np.abs(s20[name][1] - e20))
        p = np.log2(m10 / m20)
        lines.append(f"{label[name]} & {m20:.2e} & {p:.2f} & {m.ORDER[name]} \\\\")
    body = (r"\begin{tabular}{l r r r}\hline" + "\n"
            + r"Method & max error ($h=0.1$) & observed order & nominal order \\"
            + "\n\\hline\n" + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")
    write_table("tabA_all.tex", body)

    # Figure: solution curve + error curves.
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    tf = np.linspace(a, b, 400)
    ax[0].plot(tf, yA(tf), "k-", lw=2, label="exact")
    ax[0].plot(t, sols["Euler"][1], "o--", ms=4, label="Euler")
    ax[0].plot(t, sols["RK4"][1], "s--", ms=4, label="RK4")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$y$")
    ax[0].set_title("Problem A: solution, $h=0.2$"); ax[0].legend()

    for name, st in [("Euler", "o-"), ("Midpoint", "^-"),
                     ("Heun (order 3)", "v-"), ("RK4", "s-"), ("Adams PC 4", "d-")]:
        ax[1].semilogy(t[1:], np.abs(sols[name][1][1:] - exact[1:]) + 1e-16,
                       st, ms=3, label=name)
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel("absolute error")
    ax[1].set_title("Problem A: error vs $t$"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA.png")); plt.close(fig)
    print("wrote figA.png")


# ----------------------------------------------------------------------
# Problem B
# ----------------------------------------------------------------------
def problem_B():
    a, b, alpha, N = 0.0, 2.0, 1.0, 20           # h = 0.1
    sols = solve_all(fB, fBprime, a, b, alpha, N)
    t = sols["Euler"][0]
    exact = yB(t)

    # Table: exact, Euler value+error, RK4 error  (every other point to stay compact).
    lines = []
    for i in range(0, N + 1, 2):
        row = [f"{t[i]:.1f}", f"{exact[i]:.6f}",
               f"{sols['Euler'][1][i]:.6f}", f"{abs(sols['Euler'][1][i]-exact[i]):.6f}",
               f"{abs(sols['RK4'][1][i]-exact[i]):.2e}"]
        lines.append(" & ".join(row) + r" \\")
    body = (r"\begin{tabular}{r r r r r}\hline" + "\n"
            + r"$t_i$ & $y(t_i)$ & Euler & Euler error & RK4 error \\" + "\n\\hline\n"
            + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")
    write_table("tabB_errors.tex", body)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    tf = np.linspace(a, b, 400)
    ax[0].plot(tf, yB(tf), "k-", lw=2, label="exact")
    ax[0].plot(t, sols["Euler"][1], "o--", ms=4, label="Euler")
    ax[0].plot(t, sols["RK4"][1], "s--", ms=4, label="RK4")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$y$")
    ax[0].set_title("Problem B: exponential growth, $h=0.1$"); ax[0].legend()

    ax[1].semilogy(t[1:], np.abs(sols["Euler"][1][1:] - exact[1:]) + 1e-16, "o-", ms=3, label="Euler")
    ax[1].semilogy(t[1:], np.abs(sols["RK4"][1][1:] - exact[1:]) + 1e-16, "s-", ms=3, label="RK4")
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel("absolute error")
    ax[1].set_title("Problem B: error grows with the solution"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figB.png")); plt.close(fig)
    print("wrote figB.png")


# ----------------------------------------------------------------------
# Problem C  (student-designed, stiff-ish)
# ----------------------------------------------------------------------
def problem_C():
    a, b, alpha = 0.0, 3.0, 1.0

    # Stable step: h = 0.1  (h*|lambda| = 1.0, inside Euler's stability bound h<=0.2).
    Ns = 30
    sols_s = solve_all(fC, fCprime, a, b, alpha, Ns)
    ts = sols_s["Euler"][0]
    ex_s = yC(ts)
    methods = ["Euler", "Midpoint", "Heun (order 3)", "RK4", "Adams PC 4"]
    short = {"Euler": "Euler", "Midpoint": "Midpoint", "Heun (order 3)": "Heun",
             "RK4": "RK4", "Adams PC 4": "Adams PC"}
    lines = []
    for name in methods:
        e = np.max(np.abs(sols_s[name][1] - ex_s))
        lines.append(f"{short[name]} & {e:.2e} & {m.ORDER[name]} \\\\")
    body = (r"\begin{tabular}{l r r}\hline" + "\n"
            + r"Method & max error ($h=0.1$) & order \\" + "\n\\hline\n"
            + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")
    write_table("tabC_stable.tex", body)

    # Larger step: h = 0.25  (h*lambda = -2.5).  This is OUTSIDE Euler's
    # stability region (needs -2 < h*lambda < 0) but INSIDE RK4's larger
    # real-axis region, so Euler blows up while RK4 stays stable.
    Nu = 12
    sols_u = solve_all(fC, fCprime, a, b, alpha, Nu)
    tu = sols_u["Euler"][0]

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    tf = np.linspace(a, b, 500)
    ax[0].plot(tf, yC(tf), "k-", lw=2, label="exact")
    for name, st in [("Euler", "o--"), ("RK4", "s--"), ("Adams PC 4", "d--")]:
        ax[0].plot(ts, sols_s[name][1], st, ms=3, label=name)
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$y$")
    ax[0].set_title("Problem C: $h=0.1$ (all stable)"); ax[0].legend(fontsize=8)

    ax[1].plot(tf, yC(tf), "k-", lw=2, label="exact")
    ax[1].plot(tu, sols_u["Euler"][1], "o--", ms=5, label="Euler ($h=0.25$)")
    ax[1].plot(tu, sols_u["RK4"][1], "s--", ms=5, label="RK4 ($h=0.25$)")
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel("$y$")
    ax[1].set_title("Problem C: $h=0.25$ (Euler unstable)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figC.png")); plt.close(fig)
    print("wrote figC.png")

    # Numbers quoted in the report text.
    print("  Problem C, h=0.25, Euler w(3.0) =", sols_u["Euler"][1][-1],
          " RK4 w(3.0) =", sols_u["RK4"][1][-1], " exact =", yC(3.0))


if __name__ == "__main__":
    problem_A()
    problem_B()
    problem_C()
    print("done.")
