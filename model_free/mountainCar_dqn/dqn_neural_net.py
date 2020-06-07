import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DeepQNetwork(Model):
    def __init__(self, state_dim, action_dim):
        super(DeepQNetwork, self).__init__()
        self.inputs = Input(shape=(1, state_dim))
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(16, activation='relu')
        self.v_output = Dense(action_dim, activation='linear')

    def call(self, x):
        output = self.d1(x)
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
        self.model = DeepQNetwork(self.state_dim, self.action_dim)
        self.theta = self.model.trainable_variables

        # set traning method
        self.optimizer = Adam(learning_rate)

    # update neural net with batch data
    def train_on_batch(self, states, actions, td_targets):
        if self.final_exploration < self.initial_exploration:
            self.initial_exploration -= 1.0 / self.final_exploration_frame
        with tf.GradientTape() as g:
            q_values = self.model(states)
            q_values_with_actions = tf.reduce_sum(q_values * actions, axis=1)
            loss = 0.5*((td_targets-q_values_with_actions)**2)
        g_theta = g.gradient(loss, self.model.trainable_weights)
        grad = zip(g_theta, self.model.trainable_weights)
        self.optimizer.apply_gradients(grad)

    def get_action(self, state, is_test=False):
        if self.initial_exploration >= np.random.rand() and not is_test:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.model(state)[0])


    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path+'mountainCar_dqn.h5')
