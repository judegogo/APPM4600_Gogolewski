import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, CubicSpline, PchipInterpolator

# Function to interpolate
def f(x):
    return 1 / (1 + x**2)

#Problem 1
def equispaced_nodes(n):
    return np.linspace(-5, 5, n)
#Problem 2
def chebyshev_nodes(n):
    nodes = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n)) * 5  # Scale to [-5, 5]
    return np.sort(nodes)  # Ensure strictly increasing order

# Hermite interpolation function using PCHIP
def hermite_interpolation(x, xi, yi):
    return PchipInterpolator(xi, yi)(x)

# Plots the interpolation results
def plot_interpolation(x_fine, y_fine, y_interp, xi, yi, title):
    plt.figure(figsize=(8, 6))
    plt.plot(x_fine, y_fine, label='Original Function', linewidth=2)
    plt.plot(x_fine, y_interp, label=title, linestyle='--')
    plt.plot(xi, yi, 'o', label='Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Function to interpolate
x_fine = np.linspace(-5, 5, 1000)
y_fine = f(x_fine)

n_values = [5, 10, 15, 20]
all_interpolations = {}

for n in n_values:
    #xi = equispaced_nodes(n) #Problem 1
    xi = chebyshev_nodes(n) #Problem 2
    yi = f(xi)
    
    # (a) Lagrange Interpolation
    lagrange_interp = BarycentricInterpolator(xi, yi)(x_fine)
    plot_interpolation(x_fine, y_fine, lagrange_interp, xi, yi, f'Lagrange Interpolation (n={n})')
    
    # (b) Hermite Interpolation (using PCHIP)
    hermite_interp = hermite_interpolation(x_fine, xi, yi)
    plot_interpolation(x_fine, y_fine, hermite_interp, xi, yi, f'Hermite Interpolation (n={n})')
    
    # (c) Natural Cubic Spline
    natural_spline = CubicSpline(xi, yi, bc_type='natural')(x_fine)
    plot_interpolation(x_fine, y_fine, natural_spline, xi, yi, f'Natural Cubic Spline (n={n})')
    
    # (d) Clamped Cubic Spline
    clamped_spline = CubicSpline(xi, yi, bc_type=((1, 0), (1, 0)))(x_fine)
    plot_interpolation(x_fine, y_fine, clamped_spline, xi, yi, f'Clamped Cubic Spline (n={n})')
    
    # Store results for final combined plot
    all_interpolations[n] = {
        'Lagrange': lagrange_interp,
        'Hermite': hermite_interp,
        'Natural Spline': natural_spline,
        'Clamped Spline': clamped_spline
    }

# Plot all interpolations together
for n in n_values:
    plt.figure(figsize=(8, 6))
    plt.plot(x_fine, y_fine, label='Original Function', linewidth=2)
    for method, y_interp in all_interpolations[n].items():
        plt.plot(x_fine, y_interp, linestyle='--', label=f'{method} (n={n})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Comparison of Interpolation Methods (n={n})')
    plt.legend()
    plt.grid()
    plt.show()
