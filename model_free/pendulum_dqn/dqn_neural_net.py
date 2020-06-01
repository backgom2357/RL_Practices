import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

class DeepQNetwork(Model):
    def __init__(self, action_dim):
        super(DeepQNetwork, self).__init__()
        self.d1 = Dense(64, activation=LeakyReLU(), kernel_initializer=tf.keras.initializers.he_uniform())
        self.d2 = Dense(32, activation=LeakyReLU(), kernel_initializer=tf.keras.initializers.he_uniform())
        self.d3 = Dense(16, activation=LeakyReLU(), kernel_initializer=tf.keras.initializers.he_uniform())
        self.v_output = Dense(action_dim, activation='linear')

    def call(self, inputs):
        output = self.d1(inputs)
        output = self.d2(output)
        output = self.d3(output)
        return self.v_output(output)

class DQN(object):
    def __init__(self, state_dim, action_dim, learning_rate):

        # hyperparameters
        self.learning_rate = learning_rate
        self.gradient_momentum = 0.95
        self.initial_exploration = 1.0
        self.final_exploration = 0.01
        self.final_exploration_frame = 50000

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # create deep q network
        self.model = DeepQNetwork(self.action_dim)

        # set traning method
        self.model.compile(optimizer=Adam(self.learning_rate), loss='mse')

    # update neural net with batch data
    def train_on_batch(self, states, td_targets):
        if self.final_exploration < self.initial_exploration:
            self.initial_exploration -= 1.0 / self.final_exploration_frame
        return self.model.train_on_batch(states, td_targets)

    def get_action(self, state, is_test=False):
        if self.initial_exploration >= np.random.rand() and not is_test:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.model(state)[0])


    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path+'mountainCar_dqn.h5')
