from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
from mpl_toolkits import mplot3d
from matplotlib import cm
from sympy import *
import numpy
from sympy.abc import x, y
from sympy import ordered, Matrix, hessian
from math import *
from matplotlib.ticker import MaxNLocator
from itertools import product
import autograd.numpy as np
from autograd import grad, jacobian
import sympy as smp
from sympy.abc import x,y
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import *
from numpy import *

def objective(x, y):
    return x**4 + (2/3)*x**3 + (1/2)*x**2 - 2*(x**2)*y + (4/3)*y**2

def dfdx(x, y):
    return 4*x**3 + 2*x**2 + x - 4*x*y 

def dfdy(x, y):
    return (8/3)*y - 2*x**2

def gradient(x, y):
    return array([dfdx(x, y), dfdy(x, y)])
    
       
def ArmijoLineSearch(f, xk, dk, gfk, fxk, alpha0, sigma=0.5, gamma=0.1):
    """
    Minimize over the alpha from the function f(xk + alpha*dk) and alpha > 0 is assumed to be a descent direction
    
    Parameters
    ----------
    func: A callable function that is to be minimized.
    
    xk: A current point from the function that takes the form of an array.
    
    dk: The search or descent direction usually takes the value of the first 
        derivative array of the function.
        
    gfk: The gradient of the function f(x) at the point xk that takes the form of an array 
         with dimension equal to the number of variables in the function f(x).
    
    funcxk: The value of fuction f(x) at the point xk that takes the form of a float.
    
    sigma: The value of the alpha shrinkage factor that takes the form of a float.
    
    gamma: The value to control the stopping criterion that takes the form of a float.
    
    Returns
    -------
    alpha: The value of alpha at the end of the optimization that takes the form of a scalar.
    
    funcxk_+1 = Value of the function f(x) at the new point x_k+1.
    
    """
    
    der_dot = np.dot(gfk, dk)
    fxk_alpha_dk = objective(xk[0] + alpha0*dk[0], xk[1] + alpha0*dk[1])
    
    while not fxk_alpha_dk <= fxk + gamma*alpha0*der_dot:
        """
        If f (xk + alpha*dk ) <= f (xk) + gamma*alpha*dot product of gradient and direction, 
        choose alpha_k = alpha. Otherwise, set alpha = sigma*alpha and repeat this step.
        
        """
        alpha0 = alpha0 * sigma
        fxk_alpha_dk = objective(xk[0] + alpha0*dk[0], xk[1] + alpha0*dk[1])
        
    return alpha0, fxk_alpha_dk
    
def gradient_descent(f, f_grad, init, alpha=1, tol=1e-5, max_iter=1000):
    """
    Gradient descent method for unconstraint optimization problem given a starting point x which is a element
    of real numbers. The algorithm will repeat itself accoding to the following procedure:
    
    1. Define the direction, dfk, which is the negative of the derivative or gradient of the function f(x).
    2. Using a step size strategy, choose the step length alpha using the Armijo Line Search strategy.
    3. Update the x point using the formula of x := x + alpha*direction
    
    Repeat this procedure until a stopping criterion is satisfied.
    
    Parameters
    ----------
    f: A callable function that is going to be minimized through this algorithm
    
    f_grad: The first derivative of the function f(x) in the form of a vector considering the function has
            2 variables.
            
    init: The initial value of x and y in the form of an array.
    
    alpha: The initial value of the steplength in the form of a scalar.
    
    tol: The tolerance for the l2 norm of f_grad.
    
    max_iter: The maximum number of steps this algorithm will take.
    
    Returns
    -------
    xk: The vector of the two variable (x,y) in the learning path.
    
    f(xk): The value of the objective function along the learning path.
    
    """
    
    # Initialize x, f(x), and f'(x)
    xk = init
    fk = f(xk[0], xk[1])
    gfk = f_grad(xk[0], xk[1])
    gfk_norm = np.linalg.norm(gfk)
    
    # Initialize the number of steps by saving x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    print('Initial Condition: y = {}, x = {} \n'.format(fk, xk))
    
    # Take the steps limited to the maximum iteration
    while gfk_norm > tol and num_iter < max_iter:
        
        # Determining the direction of descent
        dk = -gfk
        
        # Calculating the new x, f(x), f'(x) using Almijo Line Search strategy
        alpha, fk = ArmijoLineSearch(f, xk, dk, gfk, fk, alpha0=alpha)
        xk = xk + alpha * dk
        gfk = f_grad(xk[0], xk[1])
        gfk_norm = np.linalg.norm(gfk)
        
        # Increase the number of steps or iteration by 1 and save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
        print('Iteration: {} \t y = {}, x = {}, gradient = {:.4f}'.format(num_iter, fk, xk, gfk_norm))
    
    # Printing the final results of the algorithm
    if num_iter == max_iter:
        print('\nGradient descent method does not converge to optimum level')
    else:
        print('\nSolution: \t y = {}, x = {}'.format(fk, xk))
        
    return np.array(curve_x), np.array(curve_y)
    
def plot(xs, ys):
    x = np.arange(-2,2,0.01)
    X, Y=np.meshgrid(x,x)
    Z = X**4 + (2/3) *(X**3) +0.5*(X**2) - 2 * (X**2) * Y + (4/3) * (Y**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    plt.suptitle('Gradient Descent Method')

    ax1.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
    ax1.plot(xs[-1,0], xs[-1,1], 'ro')
    ax1.set(
        title='Path During Optimization Process',
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax1.contour(X, Y, Z)
    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.axis('square')

    ax2.plot(ys, linestyle='--', marker='o', color='orange')
    ax2.plot(len(ys)-1, ys[-1], 'ro')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(
        title='Objective Function Value During Optimization Process',
        xlabel='Iterations',
        ylabel='Objective Function Value'
    )
    ax2.legend(['Armijo line search algorithm'])

    plt.tight_layout()
    plt.show()
    
    
x0 = np.array([3,3])
x,y = gradient_descent(objective, gradient, init=x0)
plot(x,y)
