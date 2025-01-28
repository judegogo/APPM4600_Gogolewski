#The Two methods we will be using is Bisection and fixed point iteration
#Bisection:
#Bisection works by taking the midpoint of the interval and checking for a sign change.
#if there is a sign change, then the root is in that interval. 
# we then halve the interval and repeat the process until the interval is small enough.

#Fixed Point Iteration:
# this method repeats by setting x_{n+1} = f(x_n) 
# This method is repeated until we reach a low enough error

# import libraries
import numpy as np
def driver():
	# use routines
	f = lambda x: (x**2)*(x-1)
	a1 = 0.5
	b1 = 2.0
	# f = lambda x: np.sin(x)
	# a = 0.1
	# b = np.pi+0.1
	tol = 1e-7
	[astar,ier] = bisection(f,a1,b1,tol)
	print("(a)")
	print('the approximate root is',astar)
	print('the error message reads:',ier)
	print('f(astar) =', f(astar))

	
	a2 = -1.0
	b2 = 0.5
	[astar,ier] = bisection(f,a2,b2,tol)
	print("(b)")
	print('the approximate root is',astar)
	print('the error message reads:',ier)
	print('f(astar) =', f(astar))
	#not successful as there exists as the sign of this interval is always negative or 0

	a3 = -1.0
	b3 = 2.0
	[astar,ier] = bisection(f,a3,b3,tol)
	print("(c)")
	print('the approximate root is',astar)
	print('the error message reads:',ier)
	print('f(astar) =', f(astar))

	#2
	print("2")
	tol2 = 1e-5
	#a
	f2 = lambda x: (x-1)*(x-3)*(x-5)
	a21 = 0.0
	b21 = 2.4
	
	[astar,ier] = bisection(f2,a21,b21,tol2)
	print("(a)")
	print('the approximate root is',astar)
	print('the error message reads:',ier)
	print('f(astar) =', f(astar))
	
	#b
	f3 = lambda x: (x-1)**2*(x-3)
	a22 = 0.0
	b22 = 2.0

	[astar,ier] = bisection(f3,a22,b22,tol2)
	print("(b)")
	print('the approximate root is',astar)
	print('the error message reads:',ier)
	print('f(astar) =', f(astar))

	#c
	f4 = lambda x: np.sin(x)
	a23 = 0.0
	b23 = 0.1
	a24 = 0.5
	b24 = (3*np.pi)/4

	print("(c)")
	print("i")
	[astar,ier] = bisection(f4,a23,b23,tol2)
	print('the approximate root is',astar)
	print('the error message reads:',ier)
	print('f(astar) =', f(astar))
	print("ii")
	[astar,ier] = bisection(f4,a24,b24,tol2)
	print('the approximate root is',astar)
	print('the error message reads:',ier)
	print('f(astar) =', f(astar))

# define routines
def bisection(f,a,b,tol):
# Inputs:
# f,a,b - function and endpoints of initial interval
# tol - bisection stops when interval length < tol
# Returns:
# astar - approximation of root
# ier - error message
# - ier = 1 => Failed
# - ier = 0 == success
# first verify there is a root we can find in the interval
	fa = f(a)
	fb = f(b)
	if (fa*fb>0):
		ier = 1
		astar = a
		return [astar, ier]
# verify end points are not a root
	if (fa == 0):
		astar = a
		ier =0
		return [astar, ier]

	if (fb ==0):
		astar = b
		ier = 0
		return [astar, ier]
	count = 0
	d = 0.5*(a+b)
	while (abs(d-a)> tol):
		fd = f(d)
		if (fd == 0):
			astar = d
			ier = 0
			return [astar, ier]
		if (fa*fd < 0):
			b = d
		else:
			a = d
		fa = fd
		d = 0.5*(a+b)
		count = count + 1
# print('abs(d-a) = ', abs(d-a))
	astar = d
	ier = 0
	return [astar, ier]



# test functions
#f1 = lambda x: 1+0.5*np.sin(x)
# fixed point is alpha1 = 1.4987....
#f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09...
#Nmax = 100
#tol = 1e-6
# test f1 '''
#x0 = 0.0
#[xstar,ier] = fixedpt(f1,x0,tol,Nmax)
#print('the approximate fixed point is:',xstar)
#print('f1(xstar):',f1(xstar))
#print('Error message reads:',ier)
#test f2 '''
#x0 = 0.0
#[xstar,ier] = fixedpt(f2,x0,tol,Nmax)
#print('the approximate fixed point is:',xstar)
#print('f2(xstar):',f2(xstar))
#print('Error message reads:',ier)
# define routines
#/def fixedpt(f,x0,tol,Nmax):
#	''' x0 = initial guess'''
#	''' Nmax = max number of iterations'''
#	''' tol = stopping tolerance'''
#	count = 0
#	while (count <Nmax):
#		count = count +1
#		x1 = f(x0)
#		if (abs(x1-x0) <tol):
#			xstar = x1
#			ier = 0
#			return [xstar,ier]
#		x0 = x1
#	xstar = x1
#	ier = 1
#	return [xstar, ier]
driver()