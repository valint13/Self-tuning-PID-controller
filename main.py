# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 21:28:24 2025

@author: Bálint
"""

import numpy as np
import matplotlib.pyplot as plt

import src.motorsim as ms
import src.neural_network as nn
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
K_p_0 = 1         # proportional gain
K_i_0 = 0         # integrative gain
K_d_0 = 1         # derivative gain

initial_gains = [K_p_0, K_i_0, K_d_0]

# initial state
omega_0 = 0
omega_dot_0 = 0
init_vals = [omega_0, omega_dot_0]

# reference values
omega_ref = 10
omega_dot_ref = 0
ref_vals = [omega_ref, omega_dot_ref]

# simulation parameters
dt = 0.01
discount = 0.1

replay_buffer = []


system = ms.DC_motor_sys(motor_params, initial_gains, init_vals, ref_vals)

actor = nn.build_actor(statespace_size = 3, actionspace_size = 3)
nn.randomize_weights(actor)

critic = nn.build_critic(statespace_size = 3, actionspace_size = 3)
nn.randomize_weights(critic)

target_actor = actor.copy()
target_critic = critic.copy()

noise_gen = ms.OU_noise(size = 3)

#%%
noise_gen.reset()
noise = noise_gen.sample()

action = actor.predict(system.x) + noise

system.update_gains(action)
system.update_matrices()
state_transition = system.state_transition(dt = 1)

reward = ms.sys_reward(system.x, system.x_dot, system.y, system.K_p, system.K_i, system.K_d)

ms.store_transition(state_transition, action, reward, replay_buffer)

_, _, sample_reward, sample_new_states = ms.sample_replay_buffer(replay_buffer, 64)

target_Q = [0] * len(sample_new_states)

for i in range(len(sample_new_states)):
    target_actior_prediction = target_actor.predict(sample_new_states[i])
    target_critic_prediction = target_critic.predict(sample_new_states[i], target_actior_prediction)
    target_Q[i] = sample_reward[i] + discount * target_critic_prediction