# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import inv

# System of equations
def f(x, y):
    return 3*x**2 - y**2

def g(x, y):
    return 3*x*y**2 - x**3 - 1

# Matrix for iteration scheme
M = np.array([[1/6, 1/18], 
              [0, 1/6]])

# Numerical iteration on the given scheme
def iterate_system(x0, y0, tol=1e-10, max_iter=100):
    x, y = x0, y0
    errors = []
    approximations = [(x, y)]
    
    for _ in range(max_iter):
        F = np.array([f(x, y), g(x, y)])
        delta = M @ F  # Matrix-vector multiplication
        
        x_new, y_new = x - delta[0], y - delta[1]
        errors.append(np.linalg.norm([x_new - x, y_new - y]))
        approximations.append((x_new, y_new))
        
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break

        x, y = x_new, y_new

    return approximations, errors

x0, y0 = 1, 1
approximations_scheme, errors_scheme = iterate_system(x0, y0)

# Newton's Method Implementation
def Jacobian(x, y):
    return np.array([[6*x, -2*y], 
                     [3*y**2 - 3*x**2, 6*x*y]])

def newtons_method(x0, y0, tol=1e-10, max_iter=100):
    x, y = x0, y0
    errors = []
    approximations = [(x, y)]
    ErrNorm = 1
    n = 0
    while  ErrNorm > tol and n < max_iter:
        J_inv = inv(Jacobian(x, y))  # Compute inverse of Jacobian
        F = np.array([f(x, y), g(x, y)])
        delta = J_inv @ F  # Newton step
        
        x_new, y_new = x - delta[0], y - delta[1]
        errors.append(np.linalg.norm([x_new - x, y_new - y]))
        approximations.append((x_new, y_new))
        
        ErrNorm = np.linalg.norm([x_new - x, y_new - y])
        
        n += 1
        x, y = x_new, y_new

    return approximations, errors

# Apply Newton's method
approximations_newton, errors_newton = newtons_method(x0, y0)

# Ensure all lists have the same length
max_len = max(len(approximations_scheme), len(approximations_newton))

def pad_list(lst, target_length):
    return lst + [(np.nan, np.nan)] * (target_length - len(lst))

approximations_scheme = pad_list(approximations_scheme, max_len)
approximations_newton = pad_list(approximations_newton, max_len)
errors_scheme += [np.nan] * (max_len - len(errors_scheme))
errors_newton += [np.nan] * (max_len - len(errors_newton))

# Create an error table
df_errors = pd.DataFrame({
    "Iteration": np.arange(max_len),
    "Scheme x": [x for x, y in approximations_scheme],
    "Scheme y": [y for x, y in approximations_scheme],
    "Scheme Error": errors_scheme,
    "Newton x": [x for x, y in approximations_newton],
    "Newton y": [y for x, y in approximations_newton],
    "Newton Error": errors_newton
})

# Display error table
print(df_errors.to_string(index=False))

# Plot the error convergence
plt.figure(figsize=(8,6))
plt.semilogy(errors_scheme, 'o-', label="Iteration Scheme")
plt.semilogy(errors_newton, 's-', label="Newton's Method")
plt.xlabel("Iteration")
plt.ylabel("Error (Log Scale)")
plt.title("Convergence of Iteration Scheme vs. Newton's Method")
plt.legend()
plt.grid(True)
plt.show()

# Approximate solution from the iteration
solution_scheme = approximations_scheme[-1]
solution_newton = approximations_newton[-1]

solution_scheme, solution_newton

# Define the function f(x, y, z) for the ellipsoid
def f(x, y, z):
    return x**2 + 4*y**2 + 4*z**2 - 16

# Compute the gradient components
def fx(x, y, z):
    return 2*x

def fy(x, y, z):
    return 8*y

def fz(x, y, z):
    return 8*z

# Iteration scheme
def iterate_to_surface(x0, y0, z0, tol=1e-10, max_iter=100):
    x, y, z = x0, y0, z0
    errors = []
    approximations = [(x, y, z)]

    for _ in range(max_iter):
        f_val = f(x, y, z)
        grad_norm_sq = fx(x, y, z)**2 + fy(x, y, z)**2 + fz(x, y, z)**2

        if grad_norm_sq < tol:  # Prevent division by zero
            break

        d = f_val / grad_norm_sq

        x_new = x - d * fx(x, y, z)
        y_new = y - d * fy(x, y, z)
        z_new = z - d * fz(x, y, z)

        error = np.linalg.norm([x_new - x, y_new - y, z_new - z])
        errors.append(error)
        approximations.append((x_new, y_new, z_new))

        if error < tol:
            break

        x, y, z = x_new, y_new, z_new

    return approximations, errors

# Initial guess
x0, y0, z0 = 1, 1, 1

# Apply iteration scheme
approximations, errors = iterate_to_surface(x0, y0, z0)

# Convert to numpy array
approximations = np.array(approximations)

# Create a DataFrame for iteration table
df_iterations = pd.DataFrame({
    "Iteration": np.arange(len(errors) + 1),
    "x": [x for x, y, z in approximations],
    "y": [y for x, y, z in approximations],
    "z": [z for x, y, z in approximations],
    "f(x,y,z)": [f(x, y, z) for x, y, z in approximations],
    "Error": [0] + errors  # First error is 0 as initial guess is the starting point
})

# Print the iteration table
print(df_iterations.to_string(index=False))

# Plot quadratic convergence check (log-log plot of errors)
plt.figure(figsize=(8,6))
plt.loglog(errors[:-1], errors[1:], 'o-', label="Error Reduction")
plt.xlabel(r"$|x_k - x_{k-1}|$")
plt.ylabel(r"$|x_{k+1} - x_k|$")
plt.title("Quadratic Convergence Check")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()

# Output final solution and convergence behavior
solution = approximations[-1]
solution, errors
