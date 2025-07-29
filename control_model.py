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

K_p = 1     # proportional gain
K_i = 0     # integrative gain
K_d = 1     # derivative gain

K = K_t/(J*R_a)
a = (K_t*K_b + B*R_a)/(J*R_a)

#%% Transfer function representation
# N = [K*K_d, K*K_p, K*K_i]
# D = [1 + K*K_d, a + K*K_p, K*K_i]

# sys1 = ct.tf(N, D, name = "DC motor", inputs = ["U"], outputs = ["omega"])
# X0 = [[0],
#       [0]]
# T = np.arange(0, 10, 0.01)

#%% State-space representation
A = np.array([[-K*K_p/(1 + K*K_d), -K*K_i/(1 + K*K_d), a],
              [1, 0, 0],
              [K*K_p/(1 + K*K_d), K*K_i/(1 + K*K_d), -a]])
B = np.array([[0, 1 - K*K_d/(1 + K*K_d)],
              [0, 0],
              [0, K*K_d/(1 + K*K_d)]])
C = np.array([[1, 0, 0],
              [0, 0, 1]])
D = np.array([0, 0])

sys1 = ct.ss(A, B, C, D, name = "DC motor", states = ["x1","x2","x3"], inputs = ["omega_ref", "omega_dot_ref"], outputs = ["x1","x3"])
#%%
impulse = ct.impulse_response(sys1)
step = ct.step_response(sys1)

#%%
# pzp = ct.pole_zero_plot(sys1)
# rl = ct.root_locus_map(sys1)
# bode_plot = ct.bode_plot(sys1)
impulse_response_plot = plt.plot(impulse.time, impulse.states[0])
step_response_plot = plt.plot(step.time, step.states[0])
