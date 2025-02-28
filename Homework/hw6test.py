import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

def driver():
    # Define the system F(x) = 0
    def F(x):
        return np.array([x[0]**2 + x[1]**2 - 4, np.exp(x[0]) + x[1] - 1])
    
    # Define the Jacobian matrix
    def JF(x):
        return np.array([[2*x[0], 2*x[1]], [np.exp(x[0]), 1]], dtype=float)
    
    # Apply Newton's Method
    x0 = np.array([1.0, -1.0])
    tol, nmax = 1e-14, 100
    rN, rnN, nfN, nJN = newton_method_nd(F, JF, x0, tol, nmax, True)
    print("Newton's Method Solution:", rN)
    
    # Apply Lazy Newton (Chord Iteration)
    nmax = 1000
    rLN, rnLN, nfLN, nJLN = lazy_newton_method_nd(F, JF, x0, tol, nmax, True)
    
    # Apply Broyden's Method
    B0 = JF(x0)
    rB, rnB, nfB = broyden_method_nd(F, B0, x0, tol, nmax, 'fwd', True)
    
    # Plot Convergence
    plot_convergence(rnN, rN, rnB, rB, rnLN, rLN)

def plot_convergence(rnN, rN, rnB, rB, rnLN, rLN):
    plt.figure(figsize=(8,6))
    
    # Compute errors
    errN = np.linalg.norm(rnN - rN, axis=1)
    errB = np.linalg.norm(rnB - rB, axis=1)
    errLN = np.linalg.norm(rnLN - rLN, axis=1)
    
    plt.semilogy(errN, 'b-o', label='Newton')
    plt.semilogy(errB, 'g-o', label='Broyden')
    plt.semilogy(errLN, 'r-o', label='Lazy Newton')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence of Newton, Broyden, and Lazy Newton')
    plt.legend()
    plt.grid(True)
    plt.show()

def newton_method_nd(f, Jf, x0, tol, nmax, verb=False):
    x = x0.copy()
    iterates = [x]
    for n in range(nmax):
        Jx = Jf(x)
        Fx = f(x)
        step = -np.linalg.solve(Jx, Fx)
        x_new = x + step
        iterates.append(x_new)
        
        if np.linalg.norm(step) < tol or np.linalg.norm(Fx) < tol:
            return x_new, np.array(iterates), n+1, n+1
        x = x_new
    
    print("Newton did not converge after", nmax, "iterations.")
    return x, np.array(iterates), nmax, nmax

def lazy_newton_method_nd(f, Jf, x0, tol, nmax, verb=False):
    x = x0.copy()
    Jx = Jf(x)
    lu, piv = lu_factor(Jx)
    iterates = [x]
    for n in range(nmax):
        Fx = f(x)
        step = -lu_solve((lu, piv), Fx)
        x_new = x + step
        iterates.append(x_new)
        
        if np.linalg.norm(step) < tol or np.linalg.norm(Fx) < tol:
            return x_new, np.array(iterates), n+1, 1
        x = x_new
    
    print("Lazy Newton did not converge after", nmax, "iterations.")
    return x, np.array(iterates), nmax, 1

def broyden_method_nd(f, B0, x0, tol, nmax, Bmat='Id', verb=False):
    x = x0.copy()
    B = np.linalg.inv(B0) if Bmat == 'inv' else B0.copy()
    iterates = [x]
    for n in range(nmax):
        Fx = f(x)
        step = -B @ Fx
        x_new = x + step
        iterates.append(x_new)
        
        dF = f(x_new) - Fx
        if np.linalg.norm(dF) > tol:
            u = step - B @ dF
            B += np.outer(u, step) / np.dot(step, dF)
        
        if np.linalg.norm(step) < tol or np.linalg.norm(Fx) < tol:
            return x_new, np.array(iterates), n+1
        x = x_new
    
    print("Broyden did not converge after", nmax, "iterations.")
    return x, np.array(iterates), nmax

driver()
