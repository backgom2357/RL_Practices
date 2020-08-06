import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

def build_model(frame_size, action_dim, agent_history_length):
    inputs = Input(shape=(frame_size, frame_size, agent_history_length))
    conv1 = Conv2D(32, (8, 8), strides=8, activation='relu')(inputs)
    conv2 = Conv2D(64, (4, 4), strides=2, activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=1, activation='relu')(conv2)
    flatten = Flatten()(conv3)
    d1 = Dense(512, activation='relu')(flatten)
    d2 = Dense(action_dim)(d1)
    model = Model(inputs=inputs, outputs=d2)
    return model

def build_duel_dqn_model(frame_size, action_dim, agent_history_length):
    inputs = Input(shape=(frame_size, frame_size, agent_history_length))
    conv1 = Conv2D(32, (8, 8), strides=8, activation='relu')(inputs)
    conv2 = Conv2D(64, (4, 4), strides=2, activation='relu')(conv1)
    flatten = Flatten()(conv2)
    d1 = Dense(512, activation='relu')(flatten)
    d2 = Dense(512, activation='relu')(flatten)
    v = Dense(1)(d1)
    a = Dense(action_dim)(d2)
    outputs = v + (a - tf.reduce_mean(a))
    model = Model(inputs=inputs, outputs=outputs)
    return model