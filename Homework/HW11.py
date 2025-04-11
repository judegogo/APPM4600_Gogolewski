import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Function to integrate
def f(s):
    return 1 / (1 + s**2)

def f_second_derivative(s):
    return (6 * s**2 - 2) / (1 + s**2)**3

def f_fourth_derivative(s):
    return (120 * s**4 - 240 * s**2 + 24) / (1 + s**2)**5

# Composite Trapezoidal Rule
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

# Composite Simpson's Rule (requires even n)
def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

# Error analysis for choosing n
def required_n_trapezoidal(tol=1e-4):
    s_vals = np.linspace(-5, 5, 1000)
    max_f2 = np.max(np.abs(f_second_derivative(s_vals)))
    return math.ceil(np.sqrt(((10)**3 / (12 * tol)) * max_f2))

def required_n_simpson(tol=1e-4):
    s_vals = np.linspace(-5, 5, 1000)
    max_f4 = np.max(np.abs(f_fourth_derivative(s_vals)))
    n_est = math.ceil((((10)**5 / (180 * tol)) * max_f4)**(1/4))
    return n_est if n_est % 2 == 0 else n_est + 1

# Transformed function for improper integral
def transformed_f(t):
    return t * np.cos(1 / t)

# Plot and compare transformed integral
def improper_integral_simpson_plot():
    a = 0.01
    b = 1.0
    n = 4  # 5 nodes (4 intervals)

    # Simpson's Rule Approximation
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = transformed_f(x)
    S = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

    # Compare to scipy quad
    result_quad, _ = quad(transformed_f, a, b)

    print("\n=== Improper Integral Comparison (1 to ∞ of cos(x)/x^3) ===")
    print("Simpson's Rule Approximation:", S)
    print("scipy.integrate.quad Result:", result_quad)
    print("Absolute Error:", abs(S - result_quad))

    # Plot the function and nodes
    t_fine = np.linspace(a, b, 1000)
    y_fine = transformed_f(t_fine)

    plt.figure(figsize=(10, 6))
    plt.plot(t_fine, y_fine, label='f(t) = t*cos(1/t)', color='blue')
    plt.plot(x, y, 'ro', label='Simpson Nodes')
    plt.title('Function and Simpson Nodes for Improper Integral')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run and compare all methods
def compare_methods():
    a, b = -5, 5
    exact_value = 2 * math.atan(5)

    n_trap = required_n_trapezoidal()
    n_simp = required_n_simpson()

    Tn = trapezoidal_rule(f, a, b, n_trap)
    Sn = simpsons_rule(f, a, b, n_simp)

    result_default_all = quad(f, a, b, full_output=1)
    result_tol1e4_all = quad(f, a, b, epsabs=1e-4, full_output=1)

    result_default = result_default_all[0]
    info_default = result_default_all[2]

    result_tol1e4 = result_tol1e4_all[0]
    info_tol1e4 = result_tol1e4_all[2]

    print("=== Integral Approximation Comparison ===")
    print(f"Trapezoidal Rule (n = {n_trap}):", Tn)
    print(f"Simpson's Rule (n = {n_simp}):", Sn)
    print("quad (default tol = 1e-6):", result_default)
    print("quad (tol = 1e-4):", result_tol1e4)
    print("\n=== Function Evaluations ===")
    print("Function evals (quad 1e-6):", info_default['neval'])
    print("Function evals (quad 1e-4):", info_tol1e4['neval'])
    print("Function evals (Trapezoidal):", n_trap + 1)
    print("Function evals (Simpson):", n_simp + 1)

if __name__ == "__main__":
    compare_methods()
    improper_integral_simpson_plot()
