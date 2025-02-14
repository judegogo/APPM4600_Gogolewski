# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erf
from scipy.optimize import fsolve


# Constants
Ti = 20       # Initial temperature [°C]
Ts = -15      # Surface temperature [°C]
alpha = 0.138e-6  # Thermal conductivity [m²/s]
t = 60 * 24 * 3600  # 60 days in seconds

# Compute critical argument components
sqrt_alpha_t = np.sqrt(alpha * t)
denominator = 2 * sqrt_alpha_t

# Define the function f(x)
def f(x):
    return erf(x / denominator) - 0.42857  # Corrected

# Define the derivative f'(x)
def df(x):
    return (2 / (np.sqrt(np.pi) * denominator)) * np.exp(-(x / denominator) ** 2)

# Find x_bar where f(x_bar) > 0 (we'll use 1 meter as upper bound)
x_max = 1.0  # meters (chosen because f(1) > 0 based on manual calculation)

# Plot the function
x_vals = np.linspace(0, x_max, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), label='f(x)')
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.title("Root Finding Problem for Water Main Depth")
plt.xlabel("Depth x [meters]")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

################################################################################
# Newton's and Secant Methods for f(x) = x^6 - x - 1

def newton_method(f, f_prime, x0, tol=1e-10, max_iter=100):
    x = x0
    errors = []
    approximations = [x]
    
    for _ in range(max_iter):
        x_new = x - f(x) / f_prime(x)
        errors.append(abs(x_new - x))
        approximations.append(x_new)
        
        if abs(x_new - x) < tol:
            break
            
        x = x_new

    return approximations, errors

def secant_method(f, x0, x1, tol=1e-10, max_iter=100):
    errors = []
    approximations = [x0, x1]
    
    for _ in range(max_iter):
        if abs(f(x1) - f(x0)) < tol:
            break
            
        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        errors.append(abs(x_new - x1))
        approximations.append(x_new)
        
        if abs(x_new - x1) < tol:
            break
            
        x0, x1 = x1, x_new

    return approximations, errors

# Define function and derivative for root finding
def f_root(x):
    return x**6 - x - 1

def f_prime_root(x):
    return 6*x**5 - 1

# Compute roots
x0_newton = 2
x0_secant, x1_secant = 2, 1

approximations_newton, errors_newton = newton_method(f_root, f_prime_root, x0_newton)
approximations_secant, errors_secant = secant_method(f_root, x0_secant, x1_secant)

# Compute the exact root numerically
exact_root = fsolve(f_root, 2)[0]

# Create error tables
error_table_newton = np.abs(np.array(approximations_newton) - exact_root)
error_table_secant = np.abs(np.array(approximations_secant) - exact_root)

# Create a DataFrame for error tables
df_errors = pd.DataFrame({
    "Iteration": np.arange(max(len(error_table_newton), len(error_table_secant))),
    "Newton Error": np.pad(error_table_newton, (0, max(0, len(error_table_secant) - len(error_table_newton))), 'constant', constant_values=np.nan),
    "Secant Error": np.pad(error_table_secant, (0, max(0, len(error_table_newton) - len(error_table_secant))), 'constant', constant_values=np.nan)
})

# Display error table
print(df_errors)  # Print table in console
df_errors.to_csv("error_table.csv", index=False)  # Save as CSV for later inspection

# Log-Log Plot
plt.figure(figsize=(8,6))
plt.loglog(error_table_newton[:-1], error_table_newton[1:], 'o-', label="Newton's Method")
plt.loglog(error_table_secant[:-1], error_table_secant[1:], 's-', label="Secant Method")
plt.xlabel(r"$|x_k - \alpha|$")
plt.ylabel(r"$|x_{k+1} - \alpha|$")
plt.title("Convergence Order Comparison")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()
