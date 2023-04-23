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

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next, k1, k2, k3, k4


def solve_to(f, y0, t0, tn, deltat_max,tol,method):
    """Solve a first-order ODE using the Euler method with adaptive step size control.
    
    Parameters:
        f (function): A function that defines the differential equation y' = f(t, y).
        y0 (float): The initial value of y at time t0.
        t0 (float): The initial time value.
        tn (float): The final time value.
        deltat_max (float): The maximum step size.
        tol (float): The tolerance
        method (string): Euler or RK4
        
    Returns:
        Two arrays containing the time values and the corresponding values of y.
    """
    # Initialize the array of time values with the initial time
    t = [t0]
    
    # Initialize the array of y values with the initial condition
    y = [y0]
    
    # Set the initial step size to deltat_max
    deltat = deltat_max
    
    
    if method == "Euler":
        
    
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
                
    elif method == "RK4":

        # Loop until t_end is reached
        while t[-1] < tn:
            # If the next step exceeds t_end, reduce the step size
            if t[-1] + deltat > tn:
                deltat = tn - t[-1]

            # Calculate the RK4 step with the current step size
            y_next, k1, k2, k3, k4 = rk4_step(f, t[-1], y[-1], deltat)

            # Estimate the error using the difference between fourth and fifth order solutions
            error = abs((k4 - y_next) / (k4 + 1e-10))

            # If the error is smaller than the tolerance, accept the step and append the new time and y values to the arrays
            if error < tol:
                t.append(t[-1] + deltat)
                y.append(y_next)
            # Otherwise, reduce the step size and try again
            else:
                deltat *= 0.8

        
    
    # Convert the arrays to numpy arrays and return them
    return np.array(t), np.array(y)

t,y = solve_to(f, 1, 0, 5, 0.1, 0.01, 'RK4')

