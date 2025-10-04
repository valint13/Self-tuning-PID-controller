# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:42:16 2025

@author: Bálint
"""

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import SGD, Adam

import numpy as np
import random
#%% neural networks
def build_actor(statespace_size, actionspace_size):
    actor = Sequential()
    actor.add(Input(shape = (statespace_size,)))
    actor.add(Dense(400, activation = "relu"))
    actor.add(Dense(300, activation = "relu"))
    actor.add(Dense(actionspace_size, activation = "tanh"))
    return actor

def build_critic(statespace_size, actionspace_size):
    # State input
    state_input = Input(shape = (statespace_size,))
    # Hidden layer 1
    layer_1_out = Dense(400, activation = "relu")(state_input)
    # Action input
    action_input = Input(shape = (actionspace_size))
    # Combine for input to layer 2
    concatenated = Concatenate()([layer_1_out, action_input])
    # Hidden layer 2
    layer_2_out = Dense(300, activation = "relu")(concatenated)
    # Output layer: Q value
    output = Dense(1, activation = "linear")(layer_2_out)
    critic = Model(inputs = [state_input, action_input], outputs = output)
    critic.compile(optimizer = Adam(learning_rate = 10e-3), loss = "MeanSquaredError")
    return critic

#%%
def randomize_weights(network, mu = 0, sig = 0.01):
    weights = network.get_weights()
    random_weights = [np.random.normal(mu, sig, size=w.shape) for w in weights]
    network.set_weights(random_weights)
    return

def sample_replay_buffer(replay_buffer, batch_size):
    if len(replay_buffer) > batch_size:
        sample = random.sample(replay_buffer, batch_size)
    else:
        sample = replay_buffer
        
    state_batch = [elem[0] for elem in sample]
    action_batch = [elem[1] for elem in sample]
    reward_batch = [elem[2] for elem in sample]
    new_state_batch = [elem[3] for elem in sample]
    return state_batch, action_batch, reward_batch, new_state_batch

#%% test
state = [1, 2, 3]
action = [0.4, 0.6, 0]

actor = build_actor(len(state), len(action))
critic = build_critic(len(state), len(action))