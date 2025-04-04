import numpy as np
import matplotlib.pyplot as plt

# Original function
f = lambda x: np.sin(x)
# Sixth-degree Taylor polynomial
def taylor_sin6(x):
    return x - x**3 / 6 + x**5 / 120
# P^3_3(x)
def pade_3_3(x):
    return (x - (7/60)*x**3) / (1 + (1/20)*x**2)
# P^2_4(x)
def pade_2_4(x):
    return x / (1 + (1/6)*x**2 + (7/360)*x**4)
# P^4_2(x)
def pade_4_2(x):
    return (x - (7/60)*x**3) / (1 + (1/20)*x**2)

# x range
x_vals = np.linspace(-5, 5, 1000)

# Eval
y_true = f(x_vals)
y_taylor = taylor_sin6(x_vals)
y_pade_3_3 = pade_3_3(x_vals)
y_pade_2_4 = pade_2_4(x_vals)
y_pade_4_2 = pade_4_2(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_true, label='sin(x)', linewidth=2)
plt.plot(x_vals, y_taylor, '--', label='6th-degree Taylor')
plt.plot(x_vals, y_pade_3_3, '--', label='Padé P(3,3)')
plt.plot(x_vals, y_pade_2_4, '--', label='Padé P(2,4)')
plt.plot(x_vals, y_pade_4_2, '--', label='Padé P(4,2)')
plt.title('Padé Approximations vs Taylor and sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()