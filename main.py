# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 21:28:24 2025

@author: Bálint
"""

import numpy as np
import matplotlib.pyplot as plt
#%% DC motor parameters

R_a = 1     # armature resistance [Ohm]
L_a = 0     # armature inductance [H]
K_b = 0.03  # back EMF constant [V/(rad/s)]

J = 1e-4    # moment of inertia of the rotor [kg*m^2]
B = 1e-5    # viscous friction coefficient [Nm]
K_t = 0.01  # torque constant [Nm/A]

motor_params = [R_a, L_a, K_b, J, B, K_t]

K_p = 1     # proportional gain
K_i = 0     # integrative gain
K_d = 1     # derivative gain

gains = [K_p, K_i, K_d]

K = K_t/(J*R_a)
a = (K_t*K_b + B*R_a)/(J*R_a)

