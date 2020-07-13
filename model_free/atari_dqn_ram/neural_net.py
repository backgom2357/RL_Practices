import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation, MaxPooling2D

class DQN(object):
    def __init__(self, frame_dim, action_dim, agent_history_length):

        # hyperparameters
        self.learning_rate = 0.00025
        self.gradient_momentum = 0.95
        self.initial_exploration = 1.0
        self.final_exploration = 0.1
        self.final_exploration_frame = 600000
        self.epsilon = self.initial_exploration

        # action dimension
        self.action_dim = action_dim

        # create deep q network
        self.model = self.build_model(frame_dim, action_dim, agent_history_length)
        self.model.summary()

        # loss and optimizer
        self.dqn_loss = tf.keras.losses.MeanSquaredError()
        self.dqn_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def build_model(self, frame_dim, action_dim, agent_history_length):
        inputs = Input(shape=(frame_dim, frame_dim, agent_history_length))
        conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
        mp = MaxPooling2D((2,2))(conv2)
        conv3 = Conv2D(64, (3, 3), activation='relu')(mp)
        conv4 = Conv2D(64, (3, 3), activation='relu')(conv3)
        flatten = Flatten()(conv4)
        full_connect_v = Dense(512, activation='relu')(flatten)
        full_connect_a = Dense(512, activation='relu')(flatten)
        d_v = Dense(1)(full_connect_v)
        d_a = Dense(action_dim)(full_connect_a)
        print(d_v.shape, d_a.shape)
        output = d_a + (d_v - tf.reduce_mean(d_a))
        model = Model(inputs=inputs, outputs=output)
        return model

    def train(self, seqs, actions, targets):
        # epsilon decay
        if self.epsilon > self.final_exploration:
            self.epsilon -= (self.initial_exploration + self.final_exploration)/self.final_exploration_frame
        # update parameters
        with tf.GradientTape() as g:
            q_values = self.model(seqs)
            q_values_with_action = tf.reduce_sum(q_values * actions, axis=1)
            loss = self.dqn_loss(targets, q_values_with_action)
            g_theta = g.gradient(loss, self.model.trainable_weights)
        g_theta, _ = tf.clip_by_global_norm(g_theta, 10)
        self.dqn_optimizer.apply_gradients(zip(g_theta, self.model.trainable_weights))


    def get_action(self, sequence):
        if self.epsilon >= np.random.rand():
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.model(sequence)[0])

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path+'dqn_boxing.h5')