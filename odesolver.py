# -*- coding: utf-8 -*-
"""


@author: ellis
"""

import numpy as np
import matplotlib.pyplot as plt

def f(t, x):
    return x

def euler_step(f,t,y,h):
    """Compute one step of the Euler method for a first-order ODE.

Parameters:
    f (function): A function that defines the differential equation y' = f(t, y).
    t (float): The current time value.
    y (float): The current value of y.
    h (float): The step size.

Returns:
    The value of y at the next time step.
"""
    return y + h * f(t,y)

def solve_to(f, y0, t0, tn, deltat_max,tol):
    """Solve a first-order ODE using the Euler method with adaptive step size control.
    
    Parameters:
        f (function): A function that defines the differential equation y' = f(t, y).
        y0 (float): The initial value of y at time t0.
        t0 (float): The initial time value.
        tn (float): The final time value.
        deltat_max (float): The maximum step size.
        tol (float): The tolerance 
        
    Returns:
        Two arrays containing the time values and the corresponding values of y.
    """
    # Initialize the array of time values with the initial time
    t = [t0]
    
    # Initialize the array of y values with the initial condition
    y = [y0]
    
    # Set the initial step size to deltat_max
    deltat = deltat_max
    
    # Perform the Euler method until the final time value is reached
    while t[-1] < tn:
        # Compute the value of y at the next time step using a single Euler step with step size deltat
        y_next = euler_step(f, t[-1], y[-1], deltat)
        
        # Compute the value of y at the next time step using two half-sized Euler steps
        y_half = euler_step(f, t[-1], y[-1], deltat / 2)
        y_next_half = euler_step(f, t[-1] + deltat / 2, y_half, deltat / 2)
        
        # Estimate the error as the difference between the two values of y
        error = np.abs(y_next - y_next_half)
        
        # If the error is smaller than the tolerance, accept the step and append the new time and y values to the arrays
        if error < tol * deltat:
            t.append(t[-1] + deltat)
            y.append(y_next)
        # Otherwise, reduce the step size and try again
        else:
            deltat = 0.8 * deltat
    
    # Convert the arrays to numpy arrays and return them
    return np.array(t), np.array(y)

t,y = solve_to(f, 1, 0, 5, 0.1, 0.01)

plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('x')
plt.title("Solution of x'=x with Euler's method")
plt.show()