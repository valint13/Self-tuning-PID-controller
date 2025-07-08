# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 18:53:27 2025

@author: Bálint
"""

import numpy as np
import control as ct
import matplotlib.pyplot as plt
#%%
R_a = 1     # armature resistance [Ohm]
L_a = 0     # armature inductance [H]
K_b = 0.03  # back EMF constant [V/(rad/s)]

J = 1e-4    # moment of inertia of the rotor [kg*m^2]
B = 1e-5    # viscous friction coefficient [Nm]
K_t = 0.01  # torque constant [Nm/A]

K_p = 1
K_i = 0
K_d = 1

K = K_t/(J*R_a)
a = (K_t*K_b + B*R_a)/(J*R_a)

N = [K*K_d, K*K_p, K*K_i]
D = [1 + K*K_d, a + K*K_p, K*K_i]

#%%
sys1 = ct.tf(N, D, name = "DC motor", inputs = ["U"], outputs = ["omega"])
X0 = [[0],
      [0]]
T = np.arange(0, 10, 0.01)

#%%
impulse = ct.impulse_response(sys1)
step = ct.step_response(sys1)

#%%
# pzp = ct.pole_zero_plot(sys1)
# rl = ct.root_locus_map(sys1)
# bode_plot = ct.bode_plot(sys1)
impulse_response_plot = plt.plot(impulse.time, impulse.states[0])
step_response_plot = plt.plot(step.time, step.states[0])
