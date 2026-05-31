# MATH 104C — Final Project 2: Numerical Methods for PDEs

Hovan Boyajian, Spring 2026

Implements and compares finite-difference methods for partial differential
equations: the 5-point stencil with Gauss–Seidel iteration for the Laplace
equation, and the Forward-Difference, Backward-Difference, and Crank–Nicolson
methods for the heat equation. The backward and Crank–Nicolson tridiagonal
systems are solved with the Crout factorization (Algorithm 6.7).

## Layout
- `code/pde_methods.py` — solvers and helpers.
- `code/run_project.py` — generates all figures (`figures/`) and LaTeX tables (`tables/`).
- `code/check_regression.py` — sanity checks against the analytic series solution.
- `report/report.tex` — the write-up; compiled to `report/report.pdf`.

## Reproduce
```bash
cd code
python3 check_regression.py     # sanity checks
python3 run_project.py          # rebuild figures + tables
cd ../report
pdflatex report.tex && pdflatex report.tex
```

## Problems
- **A:** Laplace `u_xx + u_yy = 0` on the unit square with `u(x,1)=100x`,
  `u(1,y)=100y` and zero on the other two sides. Reference solution is the
  truncated Fourier-series solution.
- **B:** Heat equation `u_t = u_xx`, `u(0,t)=u(1,t)=0`, `u(x,0)=sin(πx)`,
  exact `e^{-π²t} sin(πx)`. Forward (stable & unstable), Backward, Crank–Nicolson.
- **C1:** Heat equation with multi-mode IC `u(x,0)=sin(πx)+½ sin(3πx)`,
  showing higher modes decay faster.
- **C2:** Poisson equation `u_xx + u_yy = f(x,y)` with manufactured solution
  `u(x,y) = sin(πx) sin(πy)`, confirming `O(h²)` convergence of the 5-point stencil.
