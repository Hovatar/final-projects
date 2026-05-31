# MATH 104C — Final Project 1 (Part I): Numerical Methods for ODEs

Hovan Boyajian, Spring 2026

Compares eight numerical methods for `y' = f(t, y)`: Euler, Taylor (order 2),
midpoint, modified Euler, Heun (order 3), classical RK4, 4-step
Adams–Bashforth, and the 4th-order Adams predictor–corrector (Adams–Moulton
corrector).

## Layout
- `code/ode_methods.py` — the eight solvers.
- `code/run_project.py` — generates all figures (`figures/`) and LaTeX tables (`tables/`).
- `code/check_regression.py` — reproduces the published textbook/demo values as a correctness check.
- `report/report.tex` — the write-up; compiled to `report/report.pdf`.

## Reproduce
```bash
cd code
python3 check_regression.py     # verify against textbook numbers
python3 run_project.py          # rebuild figures + tables
cd ../report
pdflatex report.tex && pdflatex report.tex
```

## Problems
- **A:** `y' = y - t² + 1`, `y(0)=0.5`, exact `(t+1)² - ½eᵗ`.
- **B:** `y' = 2y`, `y(0)=1`, exact `e²ᵗ` (error propagation under exponential growth).
- **C:** `y' = -10y + sin t`, `y(0)=1`, exact `(10 sin t - cos t)/101 + (102/101)e⁻¹⁰ᵗ`
  (mild stiffness → explicit-method stability limits).
