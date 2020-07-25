from neural_net import build_model
from replay_memory_ram import ReplayMemory
from config import Config
from utils import *
import numpy as np
import tensorflow as tf
import os

class Agent(Config):

    def __init__(self, env, state_dim, action_dim):
        super().__init__()

        # Environment
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Replay_memory
        self.replay_memory = ReplayMemory(self.replay_memory_size, self.state_dim)

        # Q function
        self.q = build_model(self.state_dim, self.action_dim)
        self.target_q = build_model(self.state_dim, self.action_dim)
        
        # Complie
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.q.summary()

    def get_action(self, sequence):
        if self.epsilon >= np.random.rand():
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q(sequence)[0])

    def model_train(self):
        """
        Train Q
        """
        # sample batch
        seqs, actions, rewards, next_seqs, dones = self.replay_memory.sample(self.batch_size)

        # epsilon decay
        if self.epsilon > self.final_exploration:
            self.epsilon -= (self.initial_exploration + self.final_exploration)/self.final_exploration_frame
        # update parameters
        with tf.GradientTape() as g:
            # argmax action from current q
            a_next_action = self.q(next_seqs)
            argmax_action = np.argmax(a_next_action, axis=1)
            argmax_action = tf.one_hot(argmax_action, self.action_dim)

            # calculate Q(s', a')
            target_qs = self.target_q(next_seqs)

            # Double dqn
            targets = rewards + (1 - dones) * (self.discount_factor * tf.reduce_sum(target_qs * argmax_action, axis=1))

            predicts = self.q(seqs)
            train_actions = tf.one_hot(actions, self.action_dim)
            predicts = tf.reduce_sum(predicts * train_actions, axis=1)
            loss = self.loss(targets, predicts)

        g_theta = g.gradient(loss, self.q.trainable_weights)
        self.optimizer.apply_gradients(zip(g_theta, self.q.trainable_weights))

    def train(self, episodes):

        train_ep = 0

        # repeat episode
        for e in range(episodes):

            # initialize frames, episode_reward, done
            frames, done = 0, False
            sum_q_value = 0
            episode_reward = 0
            keep_action = 0
            max_q_value = -999

            # reset env and observe initial state
            state = self.env.reset()
            state = np.reshape(state, newshape=(1,4))

            while not done:

                frames += 1
                
                # render
                self.env.render()

                # get action
                action = self.get_action(state)

                # observe next frame
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1,4))

                # store transition in replay memory
                self.replay_memory.append(state, action, reward, next_state, done)

                # wait for fill data in replay memory
                if self.replay_memory.crt_idx < self.replay_start_size and not self.replay_memory.is_full():
                    state = next_state
                    continue

                # train
                self.model_train()

                state = next_state
                
                # For logs
                q = self.q(state)
                if max_q_value < np.amax(q):
                    max_q_value = np.amax(q)
                sum_q_value += np.mean(q)

                # total reward
                episode_reward += reward

                if done:
                    train_ep += 1
                    mean_q_value = sum_q_value / frames * 4
                    if train_ep % self.target_network_update_frequency == 0:
                        self.target_q.set_weights(self.q.get_weights())
                    crt_buffer_idx = self.replay_memory.crt_idx
                    if self.replay_memory.is_full():
                        crt_buffer_idx = 'full'
                    print(self.replay_memory.crt_idx, self.replay_memory.is_full())
                    print('episode: {}, Reward: {}, Epsilon: {:.5f}, buffer size: {}, max Q-value: {:.3f} Q-value: {:.2f}'.format(train_ep,
                                                                                                                            episode_reward,
                                                                                                                            self.epsilon,
                                                                                                                            crt_buffer_idx,
                                                                                                                            max_q_value,
                                                                                                                            mean_q_value))

            if train_ep % 500 == 0:
                self.q.save_weights('./save_weights/cartpole_' + str(train_ep) + 'epi.h5')