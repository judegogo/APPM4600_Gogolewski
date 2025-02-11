import numpy as np
def driver():
# test functions
    f1 = lambda x: 1+0.5*np.sin(x)
# fixed point is alpha1 = 1.4987....
    f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09...
    f3 = lambda x: (10/(x+4))**(0.5)
    Nmax = 100
    tol = 1e-10
# test f1 '''
    x0 = 0.0
    [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)
#test f2 '''
    x0 = 0.0
    [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f2(xstar):',f2(xstar))
    print('Error message reads:',ier)
# define routines
    N = 100
    x = np.zeros((N,1))
    x0 = 1.5
    [xstar,ier] = fixedpt(f3,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f3(xstar):',f3(xstar))
    print('Error message reads:',ier)
def fixedpt(f,x0,tol,Nmax):
#''' x0 = initial guess'''
#''' Nmax = max number of iterations'''
#''' tol = stopping tolerance'''
    count = 0
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        x2 = f(x1)
        x3 = f(x2)
        x1hat = Aitkins(Nmax, tol, x0, x1, x2, f)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
# alpha = (log|p_n+1 - p_n|/log|p_n - p_n-1|)/(log|p_n - p_n-1|/log|p_n-1 - p_n-2|)


Nmax = 100
tol = 1e-10

def Aitkins(Nmax, tol, p0, p1, p2, f):
    return f(p0) - ((f(p1) - f(p0))**2)/(f(p2) - 2*f(p1) + f(p0))

driver()
   



