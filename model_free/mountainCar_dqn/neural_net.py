import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

def build_model(state_size, action_dim):
    inputs = Input(shape=(state_size))
    d1 = Dense(128, activation='relu')(inputs)
    d2 = Dense(256, activation='relu')(d1)
    d3 = Dense(action_dim)(d2)
    model = Model(inputs=inputs, outputs=d3)
    return model

def build_duel_dqn_model(state_size, action_dim):
    inputs = Input(shape=(state_size))
    d1 = Dense(128, activation='relu')(inputs)
    d2 = Dense(256, activation='relu')(d1)
    flatten = Flatten()(d1)
    d1 = Dense(64, activation='relu')(flatten)
    d2 = Dense(64, activation='relu')(flatten)
    v = Dense(1)(d1)
    a = Dense(action_dim)(d2)
    outputs = v + (a - tf.reduce_mean(a))
    model = Model(inputs=inputs, outputs=outputs)
    return model