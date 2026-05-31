"""Sanity checks: reproduce the published textbook / demo numbers."""
import numpy as np
import ode_methods as m

# Problem A:  y' = y - t^2 + 1,  y(0)=0.5,  exact y=(t+1)^2 - 0.5 e^t
fA = lambda t, y: y - t**2 + 1
yA = lambda t: (t + 1)**2 - 0.5 * np.exp(t)

# Problem B:  y' = 2y,  y(0)=1,  exact e^{2t}
fB = lambda t, y: 2 * y
yB = lambda t: np.exp(2 * t)

def idx(t, val):
    return int(np.argmin(np.abs(t - val)))

print("== Problem A, h=0.2 (N=10) ==")
t, wE = m.euler(fA, 0, 2, 0.5, 10)
t, wR = m.rk4(fA, 0, 2, 0.5, 10)
t, wMid = m.midpoint(fA, 0, 2, 0.5, 10)
t, wMod = m.modified_euler(fA, 0, 2, 0.5, 10)
t, wH = m.heun3(fA, 0, 2, 0.5, 10)
ex = yA(t)
print(f"  Euler  w(1.0) = {wE[idx(t,1.0)]:.6f}  (demo 2.458176)")
print(f"  RK4    err(1.0) = {abs(wR[idx(t,1.0)]-ex[idx(t,1.0)]):.6f}  (demo 0.000019)")
print(f"  Midpt  w(2.0) = {wMid[-1]:.7f} err {abs(wMid[-1]-ex[-1]):.7f}  (Tab5.6 5.2903695 / 0.0151025)")
print(f"  ModEu  w(2.0) = {wMod[-1]:.7f} err {abs(wMod[-1]-ex[-1]):.7f}  (Tab5.6 5.2330546 / 0.0724173)")
print(f"  Heun   w(2.0) = {wH[-1]:.7f} err {abs(wH[-1]-ex[-1]):.7f}  (Tab5.7 5.3050072 / 0.0004648)")

print("== Problem B, h=0.1 (N=20) ==")
t, wE = m.euler(fB, 0, 2, 1.0, 20)
print(f"  Euler  w(2.0) = {wE[-1]:.6f} err {abs(wE[-1]-yB(2.0)):.6f}  (demo 38.337600 / 16.260550)")

print("== Convergence orders (Problem A, max error) ==")
for name, solver in [("Euler", m.euler), ("Midpoint", m.midpoint),
                     ("Heun (order 3)", m.heun3), ("RK4", m.rk4),
                     ("Adams-Bashforth 4", m.adams_bashforth4),
                     ("Adams PC 4", m.adams_pc4)]:
    errs = []
    for N in (20, 40, 80):
        t, w = solver(fA, 0, 2, 0.5, N)
        errs.append(np.max(np.abs(w - yA(t))))
    p = np.log2(errs[0] / errs[1])
    print(f"  {name:20s} maxerr@h=.1 {errs[0]:.3e}  observed order ~ {p:.2f}  (nominal {m.ORDER[name]})")
