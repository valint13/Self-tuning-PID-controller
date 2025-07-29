# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:42:16 2025

@author: Bálint
"""

from keras.models import Sequential
from keras.layers import Input, Dense, Output
from keras.optimizers import SGD, Adam

#%% neural networks
def build_actor(statespace_size, actionspace_size):
    actor = Sequential()
    actor.add(Input(shape = (statespace_size,)))
    actor.add(Dense(400, activation = "relu"))
    actor.add(Dense(300, activation = "relu"))
    actor.add(Dense(actionspace_size, activation = "tanh"))
    return actor

def build_critic(statespace_size, actionspace_size):
    critic = Sequential()
    critic.add(Input(shape = (statespace_size, actionspace_size)))
    critic.add(Dense(400, activation = "relu"))
    critic.add(Dense(300, activation = "relu"))
    critic.add(Dense(1, activation = "linear"))
    return critic