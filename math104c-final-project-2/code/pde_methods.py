"""
pde_methods.py

Finite-difference solvers used in Final Project 2:

    - laplace_5pt_gs(...)         Laplace on a rectangle, 5-point stencil
                                  solved with Gauss-Seidel iteration.
    - laplace_5pt_sor(...)        Same, with SOR (used as a comparison).
    - poisson_5pt_gs(...)         Poisson (u_xx + u_yy = f) version of the above.
    - heat_forward(...)           Explicit Forward-Difference for u_t = u_xx.
    - heat_backward(...)          Implicit Backward-Difference (Crout solve).
    - heat_crank_nicolson(...)    Crank-Nicolson (Crout solve).
    - thomas(a, b, c, d)          Crout factorization for tridiagonal systems
                                  (Algorithm 6.7 in the textbook).
    - laplace_exact(x, y, N)      Truncated Fourier-series reference for
                                  Problem A.
"""

import numpy as np


# ----------------------------------------------------------------------
# Tridiagonal solver (Crout factorization, Algorithm 6.7)
# ----------------------------------------------------------------------
def thomas(a, b, c, d):
    """Solve a tridiagonal system with subdiagonal a, diagonal b, superdiagonal c.

    Lengths: len(a) = len(c) = n-1, len(b) = len(d) = n.

    The Crout factorization writes A = L U where L is lower bidiagonal and U is
    unit upper bidiagonal, then solves Lz = d, Ux = z.  This is exactly
    Algorithm 6.7 in Burden & Faires.
    """
    n = len(b)
    l = np.zeros(n)         # diagonal of L
    u = np.zeros(n - 1)     # superdiagonal of U
    z = np.zeros(n)
    x = np.zeros(n)

    l[0] = b[0]
    u[0] = c[0] / l[0]
    z[0] = d[0] / l[0]
    for i in range(1, n - 1):
        l[i] = b[i] - a[i - 1] * u[i - 1]
        u[i] = c[i] / l[i]
        z[i] = (d[i] - a[i - 1] * z[i - 1]) / l[i]
    l[n - 1] = b[n - 1] - a[n - 2] * u[n - 2]
    z[n - 1] = (d[n - 1] - a[n - 2] * z[n - 2]) / l[n - 1]

    x[n - 1] = z[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = z[i] - u[i] * x[i + 1]
    return x


# ----------------------------------------------------------------------
# Elliptic problem A: Laplace on [0,1]^2
# ----------------------------------------------------------------------
def _apply_boundaries(W, g_bottom, g_top, g_left, g_right, x, y):
    """Write the four Dirichlet boundaries into W (modifies in place)."""
    W[:, 0] = g_bottom(x)           # y = 0
    W[:, -1] = g_top(x)             # y = 1
    W[0, :] = g_left(y)             # x = 0
    W[-1, :] = g_right(y)           # x = 1


def laplace_5pt_gs(h, g_bottom, g_top, g_left, g_right,
                   tol=1e-8, maxit=20000):
    """Solve u_xx + u_yy = 0 on (0,1)x(0,1) with the 5-point stencil and
    Gauss-Seidel iteration.

    Returns (x, y, W, iters) where W has shape (Nx+1, Ny+1) with W[i,j] = w_{ij}
    approximating u(x_i, y_j), and iters is the iteration count.

    The 5-point stencil after dividing by h^2 (with h_x = h_y = h) gives
        4 w_{ij} - w_{i+1,j} - w_{i-1,j} - w_{i,j+1} - w_{i,j-1} = 0,
    which we relax pointwise: w_{ij} <- (w_{i+1,j}+w_{i-1,j}+w_{i,j+1}+w_{i,j-1})/4.
    """
    n = int(round(1.0 / h))
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)
    W = np.zeros((n + 1, n + 1))
    _apply_boundaries(W, g_bottom, g_top, g_left, g_right, x, y)

    for it in range(1, maxit + 1):
        diff = 0.0
        for i in range(1, n):
            for j in range(1, n):
                new = 0.25 * (W[i + 1, j] + W[i - 1, j]
                              + W[i, j + 1] + W[i, j - 1])
                d = abs(new - W[i, j])
                if d > diff:
                    diff = d
                W[i, j] = new
        if diff < tol:
            return x, y, W, it
    return x, y, W, maxit


def laplace_5pt_sor(h, g_bottom, g_top, g_left, g_right,
                    omega=None, tol=1e-8, maxit=20000):
    """Same as laplace_5pt_gs but using SOR with relaxation parameter omega.
    If omega is None, use the optimal value for the unit square
    omega_opt = 2 / (1 + sin(pi h)).
    """
    n = int(round(1.0 / h))
    if omega is None:
        omega = 2.0 / (1.0 + np.sin(np.pi * h))
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)
    W = np.zeros((n + 1, n + 1))
    _apply_boundaries(W, g_bottom, g_top, g_left, g_right, x, y)

    for it in range(1, maxit + 1):
        diff = 0.0
        for i in range(1, n):
            for j in range(1, n):
                gs = 0.25 * (W[i + 1, j] + W[i - 1, j]
                             + W[i, j + 1] + W[i, j - 1])
                new = (1.0 - omega) * W[i, j] + omega * gs
                d = abs(new - W[i, j])
                if d > diff:
                    diff = d
                W[i, j] = new
        if diff < tol:
            return x, y, W, it, omega
    return x, y, W, maxit, omega


def poisson_5pt_gs(h, f_rhs, g_bottom, g_top, g_left, g_right,
                   tol=1e-8, maxit=20000):
    """Solve u_xx + u_yy = f on (0,1)^2 with the 5-point stencil and
    Gauss-Seidel iteration.

    The stencil becomes
        w_{ij} = (w_{i+1,j}+w_{i-1,j}+w_{i,j+1}+w_{i,j-1} - h^2 f_{ij}) / 4.
    """
    n = int(round(1.0 / h))
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)
    W = np.zeros((n + 1, n + 1))
    _apply_boundaries(W, g_bottom, g_top, g_left, g_right, x, y)
    F = np.zeros_like(W)
    for i in range(n + 1):
        for j in range(n + 1):
            F[i, j] = f_rhs(x[i], y[j])

    for it in range(1, maxit + 1):
        diff = 0.0
        for i in range(1, n):
            for j in range(1, n):
                new = 0.25 * (W[i + 1, j] + W[i - 1, j]
                              + W[i, j + 1] + W[i, j - 1]
                              - h * h * F[i, j])
                d = abs(new - W[i, j])
                if d > diff:
                    diff = d
                W[i, j] = new
        if diff < tol:
            return x, y, W, it
    return x, y, W, maxit


def laplace_exact(x, y, N=80):
    """Truncated Fourier-series solution of the Problem A Laplace BVP at the
    (possibly vector-valued) coordinates x, y.

    The BCs u(0,y)=u(x,0)=0, u(x,1)=100x, u(1,y)=100y split by linearity into
        u = u1 + u2,
    with u1 nonzero only on the top edge (100x) and u2 on the right edge (100y).

    For u1:  B_n = 200 (-1)^{n+1} / (n pi),
        u1(x,y) = sum_{n=1}^N B_n * sin(n pi x) * sinh(n pi y) / sinh(n pi).
    For u2 use the same series with x and y swapped.
    """
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)
    u1 = np.zeros_like(np.add.outer(x, y))     # shape (len(x), len(y))
    u2 = np.zeros_like(u1)
    for n in range(1, N + 1):
        Bn = 200.0 * (-1.0) ** (n + 1) / (n * np.pi)
        sn = np.sinh(n * np.pi)
        u1 += Bn * np.outer(np.sin(n * np.pi * x), np.sinh(n * np.pi * y)) / sn
        u2 += Bn * np.outer(np.sinh(n * np.pi * x), np.sin(n * np.pi * y)) / sn
    return u1 + u2


# ----------------------------------------------------------------------
# Parabolic problem B: heat equation
# ----------------------------------------------------------------------
def heat_forward(h, k, T, u0, alpha=1.0):
    """Forward-Difference method for u_t = alpha u_xx on (0,1) with u(0,t)=u(1,t)=0.

    Update rule (lambda = alpha k / h^2):
        w_{i, j+1} = (1 - 2*lambda) w_{i,j}
                     + lambda (w_{i+1,j} + w_{i-1,j}).

    Returns (x, t, W) with W shape (Nx+1, Nt+1), W[:,0] = u0(x).
    """
    m = int(round(1.0 / h))
    N = int(round(T / k))
    x = np.linspace(0.0, 1.0, m + 1)
    t = np.linspace(0.0, T, N + 1)
    lam = alpha * k / h ** 2
    W = np.zeros((m + 1, N + 1))
    W[:, 0] = u0(x)
    W[0, :] = 0.0
    W[-1, :] = 0.0
    for j in range(N):
        W[1:-1, j + 1] = ((1 - 2 * lam) * W[1:-1, j]
                          + lam * (W[2:, j] + W[:-2, j]))
    return x, t, W, lam


def heat_backward(h, k, T, u0, alpha=1.0):
    """Backward-Difference method for u_t = alpha u_xx on (0,1).

    At each time step, solve the tridiagonal system
        (1 + 2 lambda) w_{i,j+1} - lambda w_{i+1,j+1} - lambda w_{i-1,j+1}
            = w_{i,j}
    for i = 1..m-1, using the Crout factorization in `thomas`.
    """
    m = int(round(1.0 / h))
    N = int(round(T / k))
    x = np.linspace(0.0, 1.0, m + 1)
    t = np.linspace(0.0, T, N + 1)
    lam = alpha * k / h ** 2
    W = np.zeros((m + 1, N + 1))
    W[:, 0] = u0(x)

    n = m - 1
    a = np.full(n - 1, -lam)
    b = np.full(n, 1.0 + 2 * lam)
    c = np.full(n - 1, -lam)
    for j in range(N):
        d = W[1:-1, j].copy()
        W[1:-1, j + 1] = thomas(a, b, c, d)
    return x, t, W, lam


def heat_crank_nicolson(h, k, T, u0, alpha=1.0):
    """Crank-Nicolson method for u_t = alpha u_xx on (0,1).

    Update rule (lambda = alpha k / h^2):
        -(lambda/2) w_{i-1,j+1} + (1 + lambda) w_{i,j+1} - (lambda/2) w_{i+1,j+1}
        =  (lambda/2) w_{i-1,j} + (1 - lambda) w_{i,j} + (lambda/2) w_{i+1,j}.

    Tridiagonal solve again via Crout (thomas).
    """
    m = int(round(1.0 / h))
    N = int(round(T / k))
    x = np.linspace(0.0, 1.0, m + 1)
    t = np.linspace(0.0, T, N + 1)
    lam = alpha * k / h ** 2
    W = np.zeros((m + 1, N + 1))
    W[:, 0] = u0(x)

    n = m - 1
    a = np.full(n - 1, -lam / 2)
    b = np.full(n, 1.0 + lam)
    c = np.full(n - 1, -lam / 2)
    for j in range(N):
        wj = W[1:-1, j]
        # right-hand side from the explicit half of the scheme
        d = (1 - lam) * wj
        d[1:] += (lam / 2) * wj[:-1]
        d[:-1] += (lam / 2) * wj[1:]
        # boundary terms (u(0,*)=u(1,*)=0 so they vanish here, but include
        # them for clarity in case of nonzero BCs)
        d[0] += (lam / 2) * (W[0, j] + W[0, j + 1])
        d[-1] += (lam / 2) * (W[-1, j] + W[-1, j + 1])
        W[1:-1, j + 1] = thomas(a, b, c, d)
    return x, t, W, lam
