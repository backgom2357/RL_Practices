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