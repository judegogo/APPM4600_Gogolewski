import numpy as np
import matplotlib.pyplot as plt

# Function to interpolate
def f(x):
    return 1 / (1 + (10 * x)**2)
#Problem 1
# Function to compute Vandermonde matrix and solve for coefficients
def vandermonde_interpolation(xi, yi):
    V = np.vander(xi, increasing=True)
    c = np.linalg.solve(V, yi)  # Solve for coefficients
    return c

# Function to evaluate polynomial
def evaluate_polynomial(c, x):
    return np.polyval(c[::-1], x)  # Reverse coefficients for polyval

#Problem 2
# Compute barycentric weights
def barycentric_weights(xi):
    n = len(xi)
    w = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] /= (xi[j] - xi[i])
    return w

def chebyshev_points(n):
    return np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
# Barycentric interpolation formula 
def barycentric_lagrange(x, xi, yi, w):
    numerator = np.zeros_like(x, dtype=np.float64)
    denominator = np.zeros_like(x, dtype=np.float64)
    exact_match = np.isclose(x[:, None], xi, atol=1e-12)  # Identify exact matches
    
    for j in range(len(xi)):
        diff = x - xi[j]
        mask = np.isclose(diff, 0, atol=1e-12)
        term = np.where(mask, 0, w[j] / diff)  # Avoid division by zero
        numerator += term * yi[j]
        denominator += term
    
    # Compute interpolation, handle exact matches
    p_x = numerator / denominator
    for i, matched in enumerate(exact_match.any(axis=1)):
        if matched:
            idx = np.argmax(exact_match[i])  # Find index of exact match
            p_x[i] = yi[idx]  # Assign exact function value
    
    return p_x

# Define range for plotting
x_fine = np.linspace(-1, 1, 1001)
y_fine = f(x_fine)

plt.figure(figsize=(8,6))


for N in range(2, 21):
    
    #xi = np.linspace(-1, 1, N) #Problem 1,2 
    xi = chebyshev_points(N) #Problem 3
    yi = f(xi)
    
    # Compute interpolating polynomial
    c = vandermonde_interpolation(xi, yi)
    
    # Evaluate polynomial on fine grid
    y_poly = evaluate_polynomial(c, x_fine)
    
    # Plot
    plt.clf()
    plt.plot(x_fine, y_fine, label='f(x)', linewidth=2)
    plt.plot(x_fine, y_poly, label=f'Interpolating Polynomial (N={N})')
    plt.plot(xi, yi, 'o', label='Data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Interpolation using Vandermonde Matrix (N={N})')
    plt.legend()
    plt.grid()
    plt.pause(2.0)  # Pause for visualization step-by-step
    
    # Stop when the max polynomial value reaches about 100
    if np.max(np.abs(y_poly)) > 100:
        break

plt.show()

for N in range(2, 21):
    #xi = np.linspace(-1, 1, N) #Problem 1,2 
    xi = chebyshev_points(N) #Problem 3
    yi = f(xi)
    
    # Compute barycentric weights
    w = barycentric_weights(xi)
    
    # Evaluate polynomial on fine grid
    y_poly = barycentric_lagrange(x_fine, xi, yi, w)
    
    # Plot
    plt.clf()
    plt.plot(x_fine, y_fine, label='f(x)', linewidth=2)
    plt.plot(x_fine, y_poly, label=f'Barycentric Interpolation (N={N})')
    plt.plot(xi, yi, 'o', label='Data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Barycentric Lagrange Interpolation (N={N})')
    plt.legend()
    plt.grid()
    plt.pause(2.0)  # Pause for visualization step-by-step
    
    # Stop when the max polynomial value reaches about 100
    if np.max(np.abs(y_poly)) > 100:
        break

plt.show()

