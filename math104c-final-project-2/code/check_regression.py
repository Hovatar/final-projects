"""Lightweight sanity checks for pde_methods.

  - The Laplace series solution satisfies the zero boundaries exactly and
    matches the 100x / 100y boundaries up to a small Gibbs overshoot at the
    truncation (interior nodes are well-resolved).
  - Refining the grid for the 5-point stencil halves the max error by 4
    (O(h^2)).
  - Crank-Nicolson reproduces the exact heat-equation decay to 4 digits at
    a moderate grid.
  - Forward-difference with lambda > 1/2 actually blows up.
"""

import numpy as np
import pde_methods as m


def check_laplace_series_interior():
    """The series should give a smooth, monotone solution in the interior."""
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 11)
    U = m.laplace_exact(x, y, N=80)
    # zero-BC edges should be exactly zero
    assert np.max(np.abs(U[:, 0])) < 1e-10
    assert np.max(np.abs(U[0, :])) < 1e-10
    # interior should be in [0, 100] (max occurs on the boundary corner)
    assert U[1:-1, 1:-1].min() >= -1e-6
    assert U[1:-1, 1:-1].max() <= 100 + 1.0
    # symmetric problem: u(x, y) = u(y, x) by construction of u1 + u2
    err_sym = np.max(np.abs(U - U.T))
    print(f"  series symmetry  max|U - U^T| = {err_sym:.2e}")
    assert err_sym < 1e-10


def check_laplace_problem_A_exact_form():
    """The Project 2 Laplace BCs (u(x,1)=100x, u(1,y)=100y, zero on the other
    two sides) are matched exactly by u(x,y) = 100 x y.  Since 100xy is
    harmonic and the 5-point stencil is exact for bilinear functions, the FD
    solution should equal 100xy at every interior node up to the iteration
    tolerance.
    """
    g0 = lambda v: np.zeros_like(v)
    g_top = lambda x: 100 * x
    g_right = lambda y: 100 * y
    for h in (1 / 4, 1 / 8, 1 / 16):
        x, y, W, it = m.laplace_5pt_gs(h, g0, g_top, g0, g_right,
                                       tol=1e-9, maxit=50000)
        Ubilin = 100.0 * np.outer(x, y)
        err = np.max(np.abs(W[1:-1, 1:-1] - Ubilin[1:-1, 1:-1]))
        print(f"  h=1/{int(round(1/h))}: GS it={it:5d}, "
              f"max|w - 100xy| interior = {err:.2e}")
        assert err < 1e-5


def check_poisson_5pt_O_h2():
    """The 5-point stencil should be O(h^2) on the Poisson problem with the
    manufactured solution u(x,y) = sin(pi x) sin(pi y),
    f(x,y) = -2 pi^2 sin(pi x) sin(pi y).
    """
    u_ex = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    f_rhs = lambda x, y: -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    zero = lambda v: np.zeros_like(v)
    errs = []
    for h in (1 / 4, 1 / 8, 1 / 16):
        x, y, W, _ = m.poisson_5pt_gs(h, f_rhs, zero, zero, zero, zero,
                                      tol=1e-10, maxit=80000)
        U = np.outer(np.sin(np.pi * x), np.sin(np.pi * y))
        errs.append(np.max(np.abs(W[1:-1, 1:-1] - U[1:-1, 1:-1])))
    print("  Poisson interior max-err:", [f"{e:.3e}" for e in errs])
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print("  successive ratios:", [f"{r:.2f}" for r in ratios])
    for r in ratios:
        assert 3.5 < r < 4.5


def check_heat_cn_decay():
    u0 = lambda x: np.sin(np.pi * x)
    h, k, T = 0.02, 0.0005, 0.2
    x, t, W, lam = m.heat_crank_nicolson(h, k, T, u0)
    Ue = np.outer(np.sin(np.pi * x), np.exp(-np.pi ** 2 * t))
    err = np.max(np.abs(W - Ue))
    print(f"  CN heat:  h={h}, k={k}, lam={lam:.3f}, max|err|={err:.3e}")
    assert err < 5e-4


def check_forward_unstable():
    u0 = lambda x: np.sin(np.pi * x)
    h, k, T = 0.1, 0.01, 0.5             # lam = 1.0 > 0.5
    x, t, W, lam = m.heat_forward(h, k, T, u0)
    final_max = np.max(np.abs(W[:, -1]))
    print(f"  forward unstable:  lam={lam:.3f}, |w|_inf at T={final_max:.2e}")
    assert final_max > 1.0      # exact value at T=0.5 is < 0.01


if __name__ == "__main__":
    print("== Laplace series sanity ==")
    check_laplace_series_interior()
    print("== Problem A: FD reproduces u = 100xy ==")
    check_laplace_problem_A_exact_form()
    print("== Poisson 5-point stencil O(h^2) ==")
    check_poisson_5pt_O_h2()
    print("== Crank-Nicolson decay ==")
    check_heat_cn_decay()
    print("== Forward-difference instability ==")
    check_forward_unstable()
    print("all checks passed.")
