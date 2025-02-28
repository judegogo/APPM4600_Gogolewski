import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():
    ############################################################################
    ############################################################################
    # Rootfinding example start. You are given F(x)=0.
#Problem 1: f(x,y) = x^2+y^2-4=0, g(x,y) = e^x +y -1 = 0
#x0 = [1,1], [1,-1], [0,0]
    #First, we define F(x) and its Jacobian.
    def F(x):
        return np.array([x[0]**2 + x[1]**2 - 4, np.exp(x[0]) + x[1] - 1])
    def JF(x):
        return np.array([[2*x[0],2*x[1]],[np.exp(x[0]),1]]);

    # Apply Newton Method:
    x0 = np.array([1.0,-1.0]); tol=1e-14; nmax=100;
    (rN,rnN,nfN,nJN) = newton_method_nd(F,JF,x0,tol,nmax,True);
    print(rN)

    # Apply Lazy Newton (chord iteration)
    nmax=1000;
    (rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F,JF,x0,tol,nmax,True);

    # Apply Broyden Method
    Bmat='fwd'; B0 = JF(x0); nmax=100;
    (rB,rnB,nfB) = broyden_method_nd(F,B0,x0,tol,nmax,Bmat,True);

    # Plots and comparisons
    numN = rnN.shape[0];
    errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.title('Newton iteration log10|r-rn|');
    plt.legend();
    plt.show();

    numB = rnB.shape[0];
    errB = np.max(np.abs(rnB[0:(numB-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.plot(np.arange(numB-1),np.log10(errB+1e-18),'g-o',label='Broyden');
    plt.title('Newton and Broyden iterations log10|r-rn|');
    plt.legend();
    plt.show();

    numLN = rnLN.shape[0];
    errLN = np.max(np.abs(rnLN[0:(numLN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.plot(np.arange(numB-1),np.log10(errB+1e-18),'g-o',label='Broyden');
    plt.plot(np.arange(numLN-1),np.log10(errLN+1e-18),'r-o',label='Lazy Newton');
    plt.title('Newton, Broyden and Lazy Newton iterations log10|r-rn|');
    plt.legend();
    plt.show();
    ############################################################################
    #Problem 2: 

     # Common parameters
    x0 = np.array([0.0, 0.0, 1.0])
    tol = 1e-14
    Nmax = 100
    
   # Newton's method
    print("\nNewton's Method:")
    (r_newton, rn_newton, nf_newton, nJ_newton) = newton_method_nd(evalF, evalJ, x0, tol, Nmax, True)
    
    # Steepest Descent
    print("\nSteepest Descent:")
    [x_sd, g_sd, ier_sd, rn_sd] = SteepestDescent(x0.copy(), tol, Nmax)
    
    # Hybrid approach: Steepest Descent (5e-2) + Newton
    print("\nHybrid Method - Phase 1 (Steepest Descent):")
    [x_hybrid, g_hybrid, ier_hybrid, rn_hybrid_sd] = SteepestDescent(x0.copy(), 5e-2, Nmax)
    print("\nHybrid Method - Phase 2 (Newton):")
    (r_hybrid, rn_hybrid_newton, nf_hybrid, nJ_hybrid) = newton_method_nd(evalF, evalJ, x_hybrid, tol, Nmax, True)
    
     # Combine hybrid iterations
    rn_hybrid = np.vstack([rn_hybrid_sd, rn_hybrid_newton])
    
    # Calculate errors for each method
    errors_newton = [norm(evalF(x)) for x in rn_newton]
    errors_sd = [norm(evalF(x)) for x in rn_sd]
    errors_hybrid = [norm(evalF(x)) for x in rn_hybrid]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(errors_newton, 'b-o', label="Newton's Method", markersize=4)
    plt.semilogy(errors_sd, 'r--s', label="Steepest Descent", markersize=4)
    plt.semilogy(errors_hybrid, 'g-.d', label="Hybrid Method", markersize=4)
    
    plt.title('Convergence Comparison')
    plt.xlabel('Iteration Number')
    plt.ylabel('log(||F(x)||)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

###########################################################
#functions:
def evalF(x):

    F = np.zeros(3)
    F[0] = x[0] +math.cos(x[0]*x[1]*x[2])-1.
    F[1] = (1.-x[0])**(0.25) + x[1] +0.05*x[2]**2 -0.15*x[2]-1
    F[2] = -x[0]**2-0.1*x[1]**2 +0.01*x[1]+x[2] -1
    return F

def evalJ(x): 

    J =np.array([[1.+x[1]*x[2]*math.sin(x[0]*x[1]*x[2]),x[0]*x[2]*math.sin(x[0]*x[1]*x[2]),x[1]*x[0]*math.sin(x[0]*x[1]*x[2])],
          [-0.25*(1-x[0])**(-0.75),1,0.1*x[2]-0.15],
          [-2*x[0],-0.2*x[1]+0.01,1]])
    return J

def evalg(x):

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

# Modified SteepestDescent to return iteration history
def SteepestDescent(x, tol, Nmax):
    rn = [x.copy()]
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)
        
        if z0 < 1e-15:
            print("Zero gradient")
            return [x, g1, 0, np.array(rn)]
            
        z /= z0
        alpha = find_optimal_step(x, z, g1)
        x -= alpha*z
        rn.append(x.copy())
        
        if abs(evalg(x) - g1) < tol:
            return [x, evalg(x), 0, np.array(rn)]
            
    print("Max iterations exceeded")
    return [x, evalg(x), 1, np.array(rn)]

def find_optimal_step(x, direction, g1):
    alpha_vals = [0, 0.5, 1]
    while evalg(x - alpha_vals[-1]*direction) >= g1:
        alpha_vals[-1] /= 2
        if alpha_vals[-1] < 1e-15:
            return 0.0
            
    g_vals = [g1, evalg(x - alpha_vals[1]*direction), evalg(x - alpha_vals[2]*direction)]
    h1 = (g_vals[1] - g_vals[0])/alpha_vals[1]
    h2 = (g_vals[2] - g_vals[1])/(alpha_vals[2] - alpha_vals[1])
    h3 = (h2 - h1)/alpha_vals[2]
    
    optimal_alpha = 0.5*(alpha_vals[1] - h1/h3)
    return min(optimal_alpha, alpha_vals[2])

##############################################################################
# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Lazy Newton method (chord iteration) in n dimensions implementation
def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn);

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn);

    n=0;
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Implementation of Broyden method. B0 can either be an approx of Jf(x0) (Bmat='fwd'),
# an approx of its inverse (Bmat='inv') or the identity (Bmat='Id')
def broyden_method_nd(f,B0,x0,tol,nmax,Bmat='Id',verb=False):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1;
    npn=1;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.12f|" % (n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        #Broyden step xn = xn -B_n\Fn
        dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        # Update xn
        xn = xn + dn;
        npn=np.linalg.norm(dn);

        ###########################################################
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        Fn1 = f(xn);
        dFn = Fn1-Fn;
        nf+=1;
        I0rn = Inapp(Bapp,Bmat,Un,Vn,dFn); #In^{-1}*(Fn+1 - Fn)
        un = dn - I0rn;                    #un = dn - In^{-1}*dFn
        cn = dn.T @ (I0rn);                # We divide un by dn^T In^{-1}*dFn
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,dn)));
        Un = np.vstack((Un,(1/cn)*un));

        n+=1;
        Fn=Fn1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)


  # run the drivers only if this is called from the command line
driver()        
