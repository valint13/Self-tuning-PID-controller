# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:38:06 2025

@author: Bálint
"""

import numpy as np
#%% definitions
class DC_motor_sys(object):
    def __init__(self, motor_params, initial_gains, ref_vals, init_vals):
        # unpack parameters, gains and values
        R_a, L_a, K_b, J, B_visc, K_t = motor_params
        K_p, K_i, K_d = initial_gains
        omega_0, omega_dot_0 = init_vals
        omega_ref, omega_dot_ref = ref_vals
        # contruct matrices and vectors
        K = K_t/(J*R_a)
        a = (K_t*K_b + B_visc*R_a)/(J*R_a)

        A = np.array([[-K*K_p/(1 + K*K_d), -K*K_i/(1 + K*K_d), a],
                      [1, 0, 0],
                      [K*K_p/(1 + K*K_d), K*K_i/(1 + K*K_d), -a]])
        B = np.array([[0, 1 - K*K_d/(1 + K*K_d)],
                      [0, 0],
                      [0, K*K_d/(1 + K*K_d)]])
        C = np.array([[1, 0, 0],
                      [0, 0, 1]])
        D = np.array([0, 0])

        x0 = np. array([[omega_ref - omega_0],
                        [0],
                        [omega_0]])
        
        x_dot_0 = np. array([[omega_dot_ref - omega_dot_0],
                        [omega_ref - omega_0],
                        [omega_dot_0]])
        
        y0 = np.array([[omega_0],
                       [omega_dot_0]])
        
        r = np.array([[omega_ref],
                       [omega_dot_ref]])
        
        # initialize variables
        self.R_a = R_a
        self.L_a = L_a
        self.K_b = K_b
        self.J = J
        self.B_visc = B_visc
        self.K_t = K_t
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.K = K
        self.a = a
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x = x0                     # system state
        self.y = y0                     # system output
        self.r = r                      # reference input
        self.x_dot = x_dot_0
        
    def get_R_a(self):
        return self.R_a
    def get_L_a(self):
        return self.L_a
    def get_K_b(self):
        return self.K_b
    def get_J(self):
        return self.J
    def get_B_visc(self):
        return self.B_visc
    def get_K_t(self):
        return self.K_t
    def get_K_p(self):
        return self.K_p
    def get_K_i(self):
        return self.K_i
    def get_K_d(self):
        return self.K_d
    def get_K(self):
        return self.K
    def get_a(self):
        return self.a
    def get_A(self):
        return self.A
    def get_B(self):
        return self.B
    def get_C(self):
        return self.C
    def get_D(self):
        return self.D
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_r(self):
        return self.r
    
    def set_R_a(self, R_a):
        self.R_a = R_a
    def set_L_a(self, L_a):
        self.L_a = L_a
    def set_K_b(self, K_b):
        self.K_b = K_b
    def set_J(self, J):
        self.J = J
    def set_B(self, B):
        self.B = B
    def set_K_t(self, K_t):
        self.K_t = K_t
    def set_K_p(self, K_p):
        self.K_p = K_p
    def set_K_i(self, K_i):
        self.K_i = K_i
    def set_K_d(self, K_d):
        self.K_d = K_d
    def set_x(self, x):
        self.x = x
    def set_r(self, r):
        self.r = r
    
    def update_gains(self, d_K_p, d_K_i, d_K_d):
        self.K_p += d_K_p
        self.K_i += d_K_i
        self.k_d += d_K_d
    
    def update_sys_matrices(self):
        self.A = np.array([[-self.K*self.K_p/(1 + self.K*self.K_d), -self.K*self.K_i/(1 + self.K*self.K_d), self.a],
                      [1, 0, 0],
                      [self.K*self.K_p/(1 + self.K*self.K_d), self.K*self.K_i/(1 + self.K*self.K_d), -self.a]])
        self.B = np.array([[0, 1 - self.K*self.K_d/(1 + self.K*self.K_d)],
                      [0, 0],
                      [0, self.K*self.K_d/(1 + self.K*self.K_d)]])
        self.C = np.array([[1, 0, 0],
                      [0, 0, 1]])
        self.D = np.array([0, 0])
    
    def state_transition(self, dt):
        # state equation
        self.x_dot = self.A @ self.x + self.B @ self.r
        # output equation
        self.y = self.C @ self.x + self.D @ self.r
        # apply timestep
        x_new = self.x + dt * self.x_dot
        self.x = x_new
        return
    

def sys_reward(k1, k2, x, x_dot, y, K_p, K_i, K_d):
    """
    Calculate the reward from the systems outputs.

    Parameters
    ----------
    y : array (2x1)
        System output.
    k1 : float
        Tracking error penalty.
    k2 : float
        Control effort penalty.

    Returns
    -------
    reward : TYPE
        DESCRIPTION.
    """
    e = y[0]
    u = K_p * e + K_i * x[1] + K_d * x_dot[0]
    reward = - k1 * e^2 - k2 * u^2
    return reward
