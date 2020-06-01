import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, LeakyReLU

class DeepQNetwork(Model):
    def __init__(self, action_dim):
        super(DeepQNetwork, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), 4, activation=LeakyReLU())
        self.conv2 = Conv2D(64, (4, 4), 2, activation=LeakyReLU())
        self.conv3 = Conv2D(64, (4, 4), 1, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.out = Dense(action_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)

class DQN(object):
    def __init__(self, action_dim):

        # hyperparameters
        self.learning_rate = 0.00025
        self.gradient_momentum = 0.95
        self.initial_exploration = 1.0
        self.final_exploration = 0.1
        self.final_exploration_frame = 100000
        self.epsilon = self.initial_exploration

        # action dimension
        self.action_dim = action_dim

        # create deep q network
        self.model = DeepQNetwork(self.action_dim)

        # loss and optimizer
        self.dqn_loss = tf.keras.losses.MeanSquaredError()
        self.dqn_optimizer = tf.optimizers.RMSprop(learning_rate=self.learning_rate, momentum=self.gradient_momentum)

    def train(self, targets, seqs, actions):
        # epsilon decay
        self.epsilon -= (self.initial_exploration + self.final_exploration)/self.final_exploration_frame
        # update parameters
        with tf.GradientTape() as g:
            q_value = self.model(seqs)
            q_value_with_action = tf.gather(tf.reshape(q_value, [-1]), tf.range(0, q_value.shape[0]) * q_value.shape[1] + actions)
            loss = self.dqn_loss(targets, q_value_with_action)
        g_loss = g.gradient(loss, self.model.trainable_weights)
        self.dqn_optimizer.apply_gradients(zip(g_loss, self.model.trainable_weights))


    def get_action(self, sequence, is_test = False):
        if self.epsilon >= np.random.rand() and not is_test:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.model(sequence)[0])


    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path+'dqn.h5')
