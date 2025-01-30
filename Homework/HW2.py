#hw2
import math
import numpy as np
import matplotlib.pyplot as plt
import random

#Problem 2

A = (1/2)*np.array([[1, 1],[1 + 10**-10, 1 - 10**-10]])

#2-norm of A
A_2norm = np.linalg.norm(A, ord=2)

#inverse of A
A_inv = np.array([[1-10**10, 10**10], [1+10**10, -10**10]])


#2-norm of A_inv
A_inv_2norm = np.linalg.norm(A_inv, ord=2)

#condition number, 2-norm
condition_number = A_2norm * A_inv_2norm

print(f"2-norm of A: {A_2norm}")
print(f"2-norm of A_inv: {A_inv_2norm}")
print(f"Condition number: {condition_number}")

#Problem 3
#part b
def P3b(x):
    y = math.e**x
    return y - 1

#value of x we aim to test
xc = 9.999999995000000*10**(-10)

#value of y for part c
yc = P3b(xc)

print(f"yc: {yc}")

#polynomial approximation
def P3d(x):
    y = x + (1/2)*x**2 + (1/6)*x**3 + (1/24)*x**4 + (1/120)*x**5 + (1/720)*x**6 + (1/5040)*x**7 + (1/40320)*x**8 + (1/362880)*x**9 + (1/3628800)*x**10
    return y

yd = P3d(xc)

print(f"yd: {yd}")
#simpler polynomial approximation
def P3e(x):
    y = x + (1/2)*x**2 + (1/6)*x**3 + (1/24)*x**4
    return y

ye = P3e(xc)

print(f"ye: {ye}")

#Problem 4
#(a) Compute S
#vector t from 0 to pi with increments of pi/30
t = np.arange(0, np.pi + np.pi/30, np.pi/30)  #includes pi
y = np.cos(t)

S = 0  
N = len(t)  # Number of elements in t

#for loop method:
for k in range(N):  
    S += t[k] * y[k]  

#integrated package method: 
Snew = np.sum(t * y)

# Print the result
print(f"The sum via for loop is: {S}")
print(f"The sum via numpy is: {Snew}")

### (b) Wavy Circles Plotting

# Function x(θ), y(θ)
def wavy_circle(theta, R, delta_r, f, p):
    x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
    y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)
    return x, y

# (b.1) Single wavy circle with given parameters
theta = np.linspace(0, 2*np.pi, 300)
R, delta_r, f, p = 1.2, 0.1, 15, 0 # Parameters

x, y = wavy_circle(theta, R, delta_r, f, p)

plt.figure(figsize=(6,6))
plt.plot(x, y, label=f'R={R}, δr={delta_r}, f={f}, p={p}')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Wavy Circle (Single Curve)")
plt.axis("equal") 
plt.legend()
plt.grid(True)
plt.show()

# (b.2) Plot 10 different wavy circles
plt.figure(figsize=(6,6))

for i in range(1, 11):  
    R = i
    delta_r = 0.05
    f = 2 + i
    p = random.uniform(0, 2*np.pi)  # Random value between 0 and 2π
    
    x, y = wavy_circle(theta, R, delta_r, f, p)
    plt.plot(x, y, label=f'R={R}, f={f}, p={p:.2f}')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Wavy Circles (Multiple Curves)")
plt.axis("equal") 
plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True)
plt.show()


