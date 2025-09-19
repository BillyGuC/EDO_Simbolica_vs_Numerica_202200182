import numpy as np
import matplotlib.pyplot as plt

# Parámetro n (de tu carnet 202200182 -> n = 2)
n = 2

# Rango de la ventana
x_min, x_max, y_min, y_max = -10, 10, -10, 10

# ======================================================
# 1) Campo direccional
# ======================================================

X, Y = np.meshgrid(np.linspace(x_min, x_max, 25), np.linspace(y_min, y_max, 25))
U = 1.0
V = (1.0 / n) * np.cos(X - n * Y)
M = np.sqrt(U**2 + V**2)
U2, V2 = U / M, V / M

plt.figure(figsize=(7,6))
plt.quiver(X, Y, U2, V2, alpha=0.7)
plt.title("Campo direccional")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()

# ======================================================
# 2) Curvas solución numéricas (usando RK4)
# ======================================================

def f(x, y):
    return (1.0 / n) * np.cos(x - n * y)

def rk4_step(func, x, y, h):
    k1 = func(x, y)
    k2 = func(x + h/2, y + h*k1/2)
    k3 = func(x + h/2, y + h*k2/2)
    k4 = func(x + h, y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_ivp(y0, x0=-10.0, x1=10.0, steps=2000):
    xs = np.linspace(x0, x1, steps)
    ys = np.zeros_like(xs)
    ys[0] = y0
    h = xs[1] - xs[0]
    for i in range(1, len(xs)):
        ys[i] = rk4_step(f, xs[i-1], ys[i-1], h)
    return xs, ys

initial_conds = [-6, -2, 0, 1.5, 4]

plt.figure(figsize=(8,6))
Xb, Yb = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
Ub = 1.0
Vb = (1.0/n) * np.cos(Xb - n*Yb)
Mb = np.sqrt(Ub**2 + Vb**2)
plt.quiver(Xb, Yb, Ub/Mb, Vb/Mb, alpha=0.5)
for y0 in initial_conds:
    xs, ys = integrate_ivp(y0, x0=x_min, x1=x_max, steps=4000)
    plt.plot(xs, ys, label=f"y({x_min})={y0}")
plt.title(f"Curvas solución numéricas (RK4), n={n}")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.legend()
plt.show()

# ======================================================
# 3) Curvas solución simbólicas dadas por la fórmula
# y(x) = (1/n) * (x + arctan(1/(x-c)))
# ======================================================

cs = [-8, -2, -0.5, 0.5, 2, 6]
xs_fine = np.linspace(-9.9, 9.9, 2000)

plt.figure(figsize=(8,6))
for c in cs:
    denom = xs_fine - c
    vals = np.where(np.abs(denom) < 1e-6, np.nan,
        (1.0/n) * (xs_fine + np.arctan(1.0 / denom)))
    plt.plot(xs_fine, vals, label=f"c={c}")
plt.plot(xs_fine, xs_fine / n, linestyle='--', label=f"y = x/{n}")
plt.title("Curvas solución simbólicas y recta y = x/n")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()

# ======================================================
# 4) Gráfico asintótico: y(x) - x/n → 0 cuando x crece
# ======================================================

c_example = 1.3
xs_ex = np.linspace(0.1, 200, 2000)
ys_ex = (1.0/n) * (xs_ex + np.arctan(1.0/(xs_ex - c_example)))

plt.figure(figsize=(8,4))
plt.plot(xs_ex, ys_ex - xs_ex / n)
plt.title(f"y(x) - x/{n} para c={c_example}")
plt.xlabel("x")
plt.ylabel(f"y(x) - x/{n}")
plt.grid(True)
plt.show()