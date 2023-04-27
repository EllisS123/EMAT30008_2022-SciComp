# -*- coding: utf-8 -*-
"""

@author: ellis
"""
from odesolver import solve_to
import numpy as np

def ode(t,state):
    dt = np.zeros_like(state)
    dt[0] = state[0]*(1-state[0])-1*state[0]*state[1]/(0.1+state[0])
    dt[1] = 0.25*state[1]*(1 - state[1]/state[0])
    return dt

def boundary_diff(system, b, t0, y0, tn, h):
    _, y = solve_to(ode, y0, t0, tn, h, 0.01, 'Euler')
    return np.array(y[-1]) - np.array(b)

def numerical_shooting(system, t0, y0, h, tn, b1, b2, tol=0.1, max_iter=100):
    """Solve a system of ODEs using numerical shooting and root finding
    
    Parameters:
        system: A function that defines the differential equations
        t0 (float): The initial time value.
        y0 (list): The list of initial values of y at time t0.
        h (float): The maximum step size.
        tn (float): The final time value.
        b1 (list): Left boundary
        b2 (list): Right boundary
        
    Returns:
        
    """
    a = b1
    b = b2
    for i in range(max_iter):
        c = [(a[j] + b[j]) / 2 for j in range(len(a))]
        diff_c = boundary_diff(system, c, t0, y0, tn, h)
        if np.all(abs(diff_c) < tol):
            return solve_to(system, y0, t0, tn, h, 0.01, 'Euler')
        diff_a = boundary_diff(system,a, t0, y0, tn,h)
        if np.all(diff_a * diff_c < 0):
            b = c
        else:
            a = c
    raise ValueError('Maximum number of iterations exceeded')
    
t,state = numerical_shooting(ode,0, [2,5], 0.25, 5, [3,1], [2,0.5])