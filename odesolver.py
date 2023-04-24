# -*- coding: utf-8 -*-
"""


@author: ellis
"""

import numpy as np
import matplotlib.pyplot as plt

def system(t, y):
    dydt = np.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = -y[0]
    return dydt



def euler_step(f,t,y,h):
    dydt = system(t, y)
    y_new = [y[i] + h * dydt[i] for i in range(len(y))]
    return y_new






   

def rk4_step(f, t, y, h):
    n = len(y)
    k1 = f(t, y)
    k2 = f(t + h/2, [y[i] + h/2 * k1[i] for i in range(n)])
    k3 = f(t +h/2, [y[i] + h/2 * k2[i] for i in range(n)])
    k4 = f(t + h, [y[i] + h * k3[i] for i in range(n)])
    y_next = [y[i] + h/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(n)]
    return y_next, k1, k2, k3, k4


def solve_to(system, y0, t0, tn, deltat_max,tol,method):
    """Solve a first-order ODE using the Euler method with adaptive step size control.
    
    Parameters:
        system: A function that defines the differential equations
        y0 (list): The list of initial values of y at time t0.
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
            y_next = euler_step(system, t[-1], y[-1], deltat)
            
            # Compute the value of y at the next time step using two half-sized Euler steps
            y_half = euler_step(system, t[-1], y[-1], deltat / 2)
            y_next_half = euler_step(system, t[-1] + deltat / 2, y_half, deltat / 2)

            # Estimate the error as the difference between the two values of y
            error = max(np.abs(np.array(y_next) - np.array(y_next_half)))
            # If the error is smaller than the tolerance, accept the step and append the new time and y values to the arrays
            if error < tol * deltat:
                t.append(t[-1] + deltat)
                y.append(y_next)
            # Otherwise, reduce the step size and try again
            else:
                deltat = 0.8 * deltat
            # Append the new time and y values to the arrays
            t.append(t[-1] + deltat)
            y.append(y_next)

                
    elif method == "RK4":

        # Loop until t_end is reached
        while t[-1] < tn:
            # If the next step exceeds t_end, reduce the step size
            #if t[-1] + deltat > tn:
             #   deltat = tn - t[-1]

            # Calculate the RK4 step with the current step size
            y_next, k1, k2, k3, k4 = rk4_step(system, t[-1], y[-1], deltat)

            # Estimate the error using the difference between fourth and fifth order solutions
            error = abs((k4 - y_next) / (k4 + tol))

            # If the error is smaller than the tolerance, accept the step and append the new time and y values to the arrays
            if np.all(error < tol):
                t.append(t[-1] + deltat)
                y.append(y_next)
            # Otherwise, reduce the step size and try again
            else:
                deltat *= 0.8

        # Append the new time and y values to the arrays
        t.append(t[-1] + deltat)
        y.append(y_next)
        
    return t, y


t,state = solve_to(system, [1,-1], 0, 5, 0.01, 0.1, 'Euler')
#y_next, k1, k2, k3, k4 = rk4_step(system, 0,[1,1] , 0.1)
