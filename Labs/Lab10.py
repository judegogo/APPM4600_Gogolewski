import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

def eval_legendre(n,x):
# This subroutine evaluates the Legendre polynomials at x that are needed
# by calling your code from prelab
  p = [None]*(n+1)
  p[0] = lambda x: 1.0
  if n>0:
    p[1] = lambda x: x
  for j in range(2,n+1):
    p[j] = lambda x: (2*j-1)/j*x*p[j-1](x) - (j-1)/j*p[j-2](x)
  return p

def eval_chebyshev(n, x):
    # This subroutine evaluates the Chebyshev polynomials at x that are needed
    p = [None] * (n + 1)
    p[0] = lambda x: 1.0
    if n > 0:
        p[1] = lambda x: x
    for j in range(2, n + 1):
        p[j] = (lambda j: lambda x: (2 * j - 1) / j * x * p[j - 1](x) - (j - 1) / j * p[j - 2](x))(j)
    return p

def driver():

#  function you want to approximate
    f = lambda x: 1/(1+x**2)


# Interval of interest    
    a = -1
    b = 1
# weight function 
#   w = lambda x: 1 #Legendre
    w = lambda x: 1 / np.sqrt(1-x**2) #Chebyshev

# order of approximation
    n = 2

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
      #pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      pval[kk] = eval_chebyshev_expansion(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'r-', label= 'f(x)')
    plt.plot(xeval,pval,'b--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'r--',label='error')
    plt.legend()
    plt.show()
    
      
    

def eval_legendre_expansion(f,a,b,w,n,x): 

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab 
  p = eval_legendre(n,x)
  # initialize the sum to 0 
  pval = 0.0    
  for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: p[j](x)
      # make a function handle for evaluating phi_j^2(x)*w(x)
      phi_j_sq = lambda x: phi_j(x)**2*w(x)
      # use the quad function from scipy to evaluate normalizations
      norm_fac,err = quad(phi_j_sq,a,b)
      # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
      func_j = lambda x: f(x)*phi_j(x)*w(x)/norm_fac
      # use the quad function from scipy to evaluate coeffs
      aj,err = quad(func_j,a,b)
      # accumulate into pval
      pval = pval+aj*p[j](x)


  return pval

def eval_chebyshev_expansion(f,a,b,w,n,x):
#   This subroutine evaluates the Chebys
#  Evaluate all the Chebyshev polynomials at x that are needed

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab 
    p = eval_chebyshev(n,x)
  # initialize the sum to 0 
    pval = 0.0    
    for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: p[j](x)
      # make a function handle for evaluating phi_j^2(x)*w(x)
      phi_j_sq = lambda x: phi_j(x)**2*w(x)
      # use the quad function from scipy to evaluate normalizations
      norm_fac,err = quad(phi_j_sq,a,b)
      # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
      func_j = lambda x: f(x)*phi_j(x)*w(x)/norm_fac
      # use the quad function from scipy to evaluate coeffs
      aj,err = quad(func_j,a,b)
      # accumulate into pval
      pval = pval+aj*p[j](x)

    return pval    

driver()         
