"""
ode_methods.py
Numerical methods for first-order initial value problems  y' = f(t, y),  y(a) = alpha.

Final Project 1 (Part I) -- Numerical Methods for ODEs
PSTAT/MATH 104C
Hovan Boyajian

All step formulas follow the conventions used in lecture (Burden & Faires):
  - one-step methods: Euler, Taylor order 2, Midpoint, Modified Euler, Heun (order 3), RK4
  - multistep methods: Adams-Bashforth 4-step (explicit) and the 4th-order
    Adams predictor-corrector, whose corrector is the 3-step Adams-Moulton formula.

Each solver returns two numpy arrays (t, w) of length N+1.
"""

import numpy as np


def _grid(a, b, N):
    h = (b - a) / N
    t = a + h * np.arange(N + 1)
    return h, t


# ----------------------------------------------------------------------
# One-step methods
# ----------------------------------------------------------------------

def euler(f, a, b, alpha, N):
    """Euler's method.  Local truncation error O(h), global error O(h)."""
    h, t = _grid(a, b, N)
    w = np.empty(N + 1)
    w[0] = alpha
    for i in range(N):
        w[i + 1] = w[i] + h * f(t[i], w[i])
    return t, w


def taylor2(f, fprime, a, b, alpha, N):
    """Taylor's method of order two.

    w_{i+1} = w_i + h*f + (h^2/2)*f'(t,y),  where  f' = f_t + f_y * f.
    `fprime` must return f'(t, y).  Global error O(h^2).
    """
    h, t = _grid(a, b, N)
    w = np.empty(N + 1)
    w[0] = alpha
    for i in range(N):
        w[i + 1] = w[i] + h * f(t[i], w[i]) + 0.5 * h * h * fprime(t[i], w[i])
    return t, w


def midpoint(f, a, b, alpha, N):
    """Midpoint method (a second-order Runge-Kutta method)."""
    h, t = _grid(a, b, N)
    w = np.empty(N + 1)
    w[0] = alpha
    for i in range(N):
        k = f(t[i], w[i])
        w[i + 1] = w[i] + h * f(t[i] + h / 2, w[i] + (h / 2) * k)
    return t, w


def modified_euler(f, a, b, alpha, N):
    """Modified Euler method (the explicit trapezoidal RK2 method)."""
    h, t = _grid(a, b, N)
    w = np.empty(N + 1)
    w[0] = alpha
    for i in range(N):
        k1 = f(t[i], w[i])
        k2 = f(t[i + 1], w[i] + h * k1)
        w[i + 1] = w[i] + (h / 2) * (k1 + k2)
    return t, w


def heun3(f, a, b, alpha, N):
    """Heun's third-order method."""
    h, t = _grid(a, b, N)
    w = np.empty(N + 1)
    w[0] = alpha
    for i in range(N):
        k1 = f(t[i], w[i])
        k2 = f(t[i] + h / 3, w[i] + (h / 3) * k1)
        k3 = f(t[i] + 2 * h / 3, w[i] + (2 * h / 3) * k2)
        w[i + 1] = w[i] + (h / 4) * (k1 + 3 * k3)
    return t, w


def rk4(f, a, b, alpha, N):
    """Classical fourth-order Runge-Kutta method.  Global error O(h^4)."""
    h, t = _grid(a, b, N)
    w = np.empty(N + 1)
    w[0] = alpha
    for i in range(N):
        k1 = h * f(t[i], w[i])
        k2 = h * f(t[i] + h / 2, w[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, w[i] + k2 / 2)
        k4 = h * f(t[i + 1], w[i] + k3)
        w[i + 1] = w[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, w


# ----------------------------------------------------------------------
# Multistep methods
# ----------------------------------------------------------------------

def adams_bashforth4(f, a, b, alpha, N):
    """Explicit 4-step Adams-Bashforth method.  Global error O(h^4).

    w_{i+1} = w_i + (h/24)*(55 f_i - 59 f_{i-1} + 37 f_{i-2} - 9 f_{i-3}).
    The three starting values w1, w2, w3 are generated with RK4.
    """
    h, t = _grid(a, b, N)
    w = np.empty(N + 1)
    # Start-up with RK4 over the first three steps.
    _, w_start = rk4(f, a, a + 3 * h, alpha, 3)
    w[:4] = w_start
    fv = [f(t[j], w[j]) for j in range(4)]   # f_0, f_1, f_2, f_3
    for i in range(3, N):
        w[i + 1] = w[i] + (h / 24) * (
            55 * fv[i] - 59 * fv[i - 1] + 37 * fv[i - 2] - 9 * fv[i - 3]
        )
        fv.append(f(t[i + 1], w[i + 1]))
    return t, w


def adams_pc4(f, a, b, alpha, N):
    """Fourth-order Adams predictor-corrector method (Burden, Algorithm 5.4).

    Predictor: 4-step Adams-Bashforth.
    Corrector: 3-step Adams-Moulton,
       w_{i+1} = w_i + (h/24)*(9 f_{i+1} + 19 f_i - 5 f_{i-1} + f_{i-2}),
    evaluated once (PECE) using the predicted value inside f_{i+1}.
    Starting values w1, w2, w3 come from RK4.
    """
    h, t = _grid(a, b, N)
    w = np.empty(N + 1)
    _, w_start = rk4(f, a, a + 3 * h, alpha, 3)
    w[:4] = w_start
    fv = [f(t[j], w[j]) for j in range(4)]
    for i in range(3, N):
        # Predict with Adams-Bashforth.
        wp = w[i] + (h / 24) * (
            55 * fv[i] - 59 * fv[i - 1] + 37 * fv[i - 2] - 9 * fv[i - 3]
        )
        # Correct with Adams-Moulton (one application).
        fp = f(t[i + 1], wp)
        w[i + 1] = w[i] + (h / 24) * (
            9 * fp + 19 * fv[i] - 5 * fv[i - 1] + fv[i - 2]
        )
        fv.append(f(t[i + 1], w[i + 1]))
    return t, w


# Registry used by the driver.  The Taylor method is handled separately
# because it needs the analytic derivative f'.
ONE_STEP = {
    "Euler": euler,
    "Midpoint": midpoint,
    "Modified Euler": modified_euler,
    "Heun (order 3)": heun3,
    "RK4": rk4,
}

MULTISTEP = {
    "Adams-Bashforth 4": adams_bashforth4,
    "Adams PC 4": adams_pc4,
}

# Nominal global order of accuracy for each method (used in discussion tables).
ORDER = {
    "Euler": 1,
    "Taylor 2": 2,
    "Midpoint": 2,
    "Modified Euler": 2,
    "Heun (order 3)": 3,
    "RK4": 4,
    "Adams-Bashforth 4": 4,
    "Adams PC 4": 4,
}
