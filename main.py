# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 21:28:24 2025

@author: Bálint
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
dt = 0.01       # timestep
discount = 0.1  # discount rate for target Q-value
tau = 0.001     # coefficient for the soft update of target networks

replay_buffer = []


system = ms.DC_motor_sys(motor_params, initial_gains, init_vals, ref_vals)

actor = nn.build_actor(statespace_size = 3, actionspace_size = 3)
nn.randomize_weights(actor)

critic = nn.build_critic(statespace_size = 3, actionspace_size = 3)
nn.randomize_weights(critic)

target_actor = actor.copy()
target_critic = critic.copy()

noise_gen = ms.OU_noise_generator(size = 3)

#%%
# generate random noise
noise_gen.reset()
noise = noise_gen.sample()

# add random noise to action
action = actor.predict(system.x) + noise

# apply action to system
system.update_gains(action)
system.update_matrices()
state_transition = system.state_transition(dt = 1)
reward = ms.sys_reward(system.x, system.x_dot, system.y, system.K_p, system.K_i, system.K_d)
ms.store_transition(state_transition, action, reward, replay_buffer)

# sample replay buffer
state_batch, action_batch, reward_batch, new_state_batch = ms.sample_replay_buffer(replay_buffer, 64)

# initialize target_Q variable
target_Q_batch = [0] * len(new_state_batch)

# calculate target Q values
for i in range(len(new_state_batch)):
    target_actor_prediction = target_actor.predict(new_state_batch[i])
    target_critic_prediction = target_critic.predict(new_state_batch[i], target_actor_prediction)
    target_Q_batch[i] = reward_batch[i] + discount * target_critic_prediction

# update critic network weights
critic.train_on_batch([state_batch, action_batch], target_Q_batch)

# make forward pass for gradient tracking
with tf.GradientTape() as tape:
    # perform forward pass on the actor network for gradient tracking
    actor_prediction = actor(state_batch, training=True)
    # perform forward pass on the critic network for gradient tracking
    critic_prediciton = critic([state_batch, actor_prediction], training=True)

# compute policy gradient
policy_gradient = tape.gradient(critic_prediciton, actor.trainable_variables)

# initialize an Adam optimizer for the actor gradient update
actor_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)

# apply the updated gradients to the actor network
actor_optimizer.apply_gradients(zip(-policy_gradient, actor.trainable_variables))

# soft updates on target networks
target_actor.set_weights(tau * actor.get_weights() + (1 - tau) * target_actor.get_weights())
target_critic.set_weights(tau * actor.get_weights() + (1 - tau) * target_actor.get_weights())
