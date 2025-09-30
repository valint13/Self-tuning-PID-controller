# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 21:28:24 2025

@author: Bálint
"""

import numpy as np
import matplotlib.pyplot as plt

import src.motorsim as ms
#%% DC motor parameters
# system parameters
R_a = 1         # armature resistance [Ohm]
L_a = 0         # armature inductance [H]
K_b = 0.03      # back EMF constant [V/(rad/s)]

J = 1e-4        # moment of inertia of the rotor [kg*m^2]
B_visc = 1e-5   # viscous friction coefficient [Nm]
K_t = 0.01      # torque constant [Nm/A]

motor_params = [R_a, L_a, K_b, J, B_visc, K_t]

# initial conditions
# initial gains
K_p = 1         # proportional gain
K_i = 0         # integrative gain
K_d = 1         # derivative gain

initial_gains = [K_p, K_i, K_d]

# initial state
omega_0 = 0
omega_dot_0 = 0
init_vals = [omega_0, omega_dot_0]

# reference values
omega_ref = 10
omega_dot_ref = 0
ref_vals = [omega_ref, omega_dot_ref]




system = ms.DC_motor_sys(motor_params, initial_gains, init_vals, ref_vals)