# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;

def fun(x):
    return np.exp(x**2+7*x+10)-1;
def dfun(x):
    return 2*x*np.exp(x**2+7*x+10)+7*np.exp(x**2+7*x+10);
def ddfun(x):
    return 2*np.exp(x**2+7*x+10)+4*x*np.exp(x**2+7*x+10)+49*np.exp(x**2+7*x+10);
def g(x):
    return x-fun(x)/dfun(x);
def dg(x):
    return 1-(dfun(x)*dfun(x)-ddfun(x)*fun(x))/(dfun(x)*dfun(x));
x = np.arange(0,np.pi,0.01);
y = fun(x);
plt.plot(x,y,x,0*x);
plt.xlabel('x'); plt.ylabel('y=f(x)');
plt.show();
input(); #pauses until user input
def bisect_method(f,a,b,tol,nmax,vrb=False):
#Bisection method applied to f between a and b
# Initial values for interval [an,bn], midpoint xn
    an = a; bn=b; n=0;
    xn = (an+bn)/2;
    aprime = dfun(a);
    bprime = dfun(b);
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
        if (dg(an)<10**-3 or dg(bn)<10**-3):
            return r, rn;
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

#(2) We will use Bisection as an algorithm to find us the basin of convergence which we will then use in Newtons Method.
#(3) The changes we need to make to bisection are to have bisection regularly check if the derivative of our 
#function is less than 1, when the derivative is less than 1 on the bounds then we can break and we now have our basin
#This parameter will be used to determine the basin of convergence,
#which will be our bounds for Newtons method. Newtons Bounds = [r-BasinofConvergence,r+BasinofConvergence]
BasinofConvergence = dfun(x);
(r,rn)=bisect_method(fun,2,4.5,1e-3,100,True);
plt.show();
input(); #pause until user input.
en = np.abs(r-rn[0:len(rn)-1]);
c1 = np.polyfit(np.arange(0,len(en)),np.log10(en),1);
print(10**c1); # 10 to the power of this coefficient is approximately equal to 0.5

################################################################################
def newton_method(f,df,x0,tol,nmax,verb=False):
#newton method to find root of f starting at guess x0
#Initialize iterates and iterate list
    xn=x0;
    rn=np.array([x0]);
# function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)
    if abs(dfn)<dtol:
#If derivative is too small, Newton will fail. Error message is
#displayed and code terminates.
        if verb:
            print('\n derivative at initial guess is near 0, try different x0 \n');
        else:
            n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|");
#Iteration runs until f(xn) is small enough or nmax iterations are computed.
        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)));
    
            pn = - fn/dfn; #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;
#Update guess adding Newton step
            xn = xn + pn;
# Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            dfn=df(xn);
            fn=f(xn);
            nfun+=2;
        r=xn;
        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));
    return (r,rn,nfun)
################################################################################
# Now, we apply this method to our test function
(r,rn,nfun)=newton_method(g,dg,r,1e-14,100,True);
# We plot n against log10|f(rn)|
plt.plot(np.arange(0,rn.size),np.log10(np.abs(fun(rn))),'r-o');
plt.xlabel('n'); plt.ylabel('log10|f(rn)|');
plt.suptitle("Newton method results");
plt.show();
input();

