# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:38:06 2025

@author: Bálint
"""

import numpy as np
#%% definitions
def compile_sys(motor_params, gains):
    # unpack parameters and gains
    R_a, L_a, K_b, J, B, K_t = motor_params
    K_p, K_i, K_d = gains
    K = K_t/(J*R_a)
    a = (K_t*K_b + B*R_a)/(J*R_a)
    
    # construct matrices and system
    A = np.array([[-K*K_p/(1 + K*K_d), -K*K_i/(1 + K*K_d), a],
                  [1, 0, 0],
                  [K*K_p/(1 + K*K_d), K*K_i/(1 + K*K_d), -a]])
    B = np.array([[0, 1 - K*K_d/(1 + K*K_d)],
                  [0, 0],
                  [0, K*K_d/(1 + K*K_d)]])
    C = np.array([[1, 0, 0],
                  [0, 0, 1]])
    D = np.array([0, 0])
    
    # compile output
    sys = sys = [A, B, C, D]
    
    return sys

def update_gains(gains, action):
    updated_gains = gains + action
    return updated_gains

def state_transition(sys, dt, u, x):
    # unpack matrices
    A, B, C, D = sys
    
    # state equation
    x_dot = A @ x + B @ u
    
    # output equation
    y = C @ x + D @ u
    
    # apply timestep
    x_new = x + dt * x_dot
    
    # compile output
    transition_result = [x_new, y]
    
    return transition_result
