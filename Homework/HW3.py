################################################################################
# This python script presents examples regarding the bisection method for
# 1D nonlinear root-finding, as presented in class.
# APPM 4650 Fall 2021
################################################################################
# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;
# Problem 1: Find the root of the function f(x) = 2x - sin(x) - 1
# Problem 2(a): (x-5)^9
# Problem 2(b): (x-5)^9 in expanded form 
# Problem 3: x^3 + x -4 
def fun(x):
    #return 2*x-np.sin(x)-1;
    #return (x-5)**9;
    #return x**9-45*x**8+900*x**7-10500*x**6+78750*x**5-393750*x**4+1312500*x**3-2812500*x**2+3515625*x-1953125
    return x**3+x-4;
# We plot our function and the x-axis y=0. We notice there is a unique intersection
# between 3 and 4.
x = np.arange(0,np.pi,0.01);
y = fun(x);
plt.plot(x,y,x,0*x);
plt.xlabel('x'); plt.ylabel('y=f(x)');
plt.show();
input(); #pauses until user input
################################################################################
# We now implement the bisection method. Note that this can also be written in a
# separate python file and imported as a module
def bisect_method(f,a,b,tol,nmax,vrb=False):
#Bisection method applied to f between a and b
# Initial values for interval [an,bn], midpoint xn
    an = a; bn=b; n=0;
    xn = (an+bn)/2;
# Current guess is stored at rn[n]
    rn=np.array([xn]);
    r=xn;
    ier=0;
    if vrb:
        print("\n Bisection method with nmax=%d and tol=%1.1e\n" % (nmax, tol));
    # The code cannot work if f(a) and f(b) have the same sign.
    # In this case, the code displays an error message, outputs empty answers and exits.
    if f(a)*f(b)>=0:
        print("\n Interval is inadequate, f(a)*f(b)>=0. Try again \n")
        print("f(a)*f(b) = %1.1f \n" % f(a)*f(b));
        r = None;
        return r
    else:
#If f(a)f(b), we proceed with the method.
        if vrb:
            print("\n|--n--|--an--|--bn--|----xn----|-|bn-an|--|---|f(xn)|---|");
        # We start two plots as subplots in the same figure.
            fig, (ax1, ax2) = plt.subplots(1, 2); #Creates figure fig and subplots
            fig.suptitle('Bisection method results'); #Sets title of the figure
            ax1.set(xlabel='x',ylabel='y=f(x)'); #Sets labels for axis for subplot 1
# We plot y=f(x) on the left subplot.
            xl=np.linspace(a,b,100,endpoint=True); yl=f(xl);
            ax1.plot(xl,yl);
    while n<=nmax:
        if vrb:
            print("|--%d--|%1.4f|%1.4f|%1.8f|%1.8f|%1.8f|" % (n,an,bn,xn,bn-
            an,np.abs(f(xn))));
################################################################
# Plot results of bisection on subplot 1 of 2 (horizontal).
            xint = np.array([an,bn]);
            yint=f(xint);
            ax1.plot(xint,yint,'ko',xn,f(xn),'rs');
################################################################
# Bisection method step: test subintervals [an,xn] and [xn,bn]
# If the estimate for the error (root-xn) is less than tol, exit
        if (bn-an)<2*tol: # better test than np.abs(f(xn))<tol
            ier=1;
            break;
# If f(an)*f(xn)<0, pick left interval, update bn
        if f(an)*f(xn)<0:
            bn=xn;
        else:
#else, pick right interval, update an
            an=xn;
# update midpoint xn, increase n.
        n += 1;
        xn = (an+bn)/2;
        rn = np.append(rn,xn);
# Set root estimate to xn.
    
    r=xn;
    if vrb:
########################################################################
# subplot 2: approximate error log-log plot
        e = np.abs(r-rn[0:n]);
#length of interval
        ln = (b-a)*np.exp2(-np.arange(0,e.size));
#log-log plot error vs interval length
        ax2.plot(-np.log2(ln),np.log2(e),'r--');
        ax2.set(xlabel='-log2(bn-an)',ylabel='log2(error)');
########################################################################
    return r, rn;
################################################################################
# Now, we run this method for our example function
# We set the interval [a,b]
# P1 [0,pi]
# P2 [4.82,5.2] tol: 1e-4
# P3 [1,4] tol: 1e-3
(r,rn)=bisect_method(fun,1,4,1e-3,100,True);
plt.show();
input(); #pause until user input.
# Finally, we can check that |r-rn| is proportional to (1/2)^n, and so,
# log10|r-rn| ~ n (log10(1/2)) + C.
en = np.abs(r-rn[0:len(rn)-1]);
c1 = np.polyfit(np.arange(0,len(en)),np.log10(en),1);
print(10**c1); # 10 to the power of this coefficient is approximately equal to 0.5

#Problem 5: 

def g(x):
    return x - 4*np.sin(2*x) - 3;

x = np.linspace(-3, 10, 1500)
plt.plot(x, g(x), label=r'$g(x) = x - 4\sin(2x) - 3$')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

#P5 (b)
#Fixed Point Iteration
# test functions
#g(x) = x - 4*np.sin(2*x) - 3
f1 = lambda x: -np.sin(2*x) +5*x/4 - 3/4
# fixed point is alpha1 = 1.4987....
f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09...
Nmax = 100
tol = 1e-6
# test f1 '''
x0 = -0.5
# define routines
def fixedpt(f,x0,tol,Nmax):
#''' x0 = initial guess'''
#''' Nmax = max number of iterations'''
#''' tol = stopping tolerance'''
    count = 0
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        
    x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]

[xstar,ier] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)
#test f2 '''
#x0 = 0.0
#[xstar,ier] = fixedpt(f2,x0,tol,Nmax)
#print('the approximate fixed point is:',xstar)
#print('f2(xstar):',f2(xstar))
#print('Error message reads:',ier)
