import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation

class DeepQNetwork(Model):

    # Duel DQN

    def __init__(self, state_dim, action_dim, agent_history_length):
        super(DeepQNetwork, self).__init__()
        self.input_state = Input(shape=(None, state_dim,state_dim, agent_history_length))
        self.conv1 = Conv2D(32, (8, 8), 4, activation='relu')
        self.conv2 = Conv2D(64, (4, 4), 2)
        self.activ = Activation('relu')
        self.flatten = Flatten()
        self.full_connect_v = Dense(512, activation='relu')
        self.full_connect_a = Dense(512, activation='relu')
        self.d_v = Dense(1)
        self.d_a = Dense(action_dim)

    def call(self, x):
        # x = self.input_state(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.flatten(x)
        v = self.full_connect_v(x)
        a = self.full_connect_a(x)
        v = self.d_v(v)
        a = self.d_a(a)
        return v, a

class DQN(object):
    def __init__(self, state_dim, action_dim, agent_history_length):

        # hyperparameters
        self.learning_rate = 0.00025
        self.gradient_momentum = 0.95
        self.initial_exploration = 1.0
        self.final_exploration = 0.1
        self.final_exploration_frame = 50000
        self.epsilon = self.initial_exploration

        # action dimension
        self.action_dim = action_dim

        # create deep q network
        self.model = DeepQNetwork(state_dim, action_dim, agent_history_length)

        # loss and optimizer
        self.dqn_loss = tf.keras.losses.MeanSquaredError()
        self.dqn_optimizer = tf.optimizers.RMSprop(learning_rate=self.learning_rate, momentum=self.gradient_momentum)

    def train(self, seqs, actions, targets):
        # epsilon decay
        if self.epsilon > self.final_exploration:
            self.epsilon -= (self.initial_exploration + self.final_exploration)/self.final_exploration_frame
        # update parameters
        with tf.GradientTape() as g:
            v, a = self.model(seqs)
            q_values = a + \
                       (v - tf.reshape(tf.reduce_mean(a, axis=1), shape=(len(a), 1)))
            q_values_with_action = tf.reduce_sum(q_values * actions, axis=1)
            loss = self.dqn_loss(targets, q_values_with_action)
        g_theta = g.gradient(loss, self.model.trainable_weights)
        g_theta, _ = tf.clip_by_global_norm(g_theta, 10)
        self.dqn_optimizer.apply_gradients(zip(g_theta, self.model.trainable_weights))


    def get_action(self, sequence):
        if self.epsilon >= np.random.rand():
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.model(sequence)[1])

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path+'dqn_boxing.h5')