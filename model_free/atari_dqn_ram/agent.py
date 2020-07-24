from neural_net import build_model
from replay_memory_ram import ReplayMemory
from config import Config
from utils import preprocess
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import wandb
wandb.init(project="dqn-pong")

class Agent(Config):

    def __init__(self, env, state_dim, action_dim):
        super().__init__(env, state_dim, action_dim)
        # Replay_memory
        self.replay_memory = ReplayMemory(self.replay_memory_size, self.frame_size, self.agent_history_length)

        # Q function
        self.q = build_model(self.frame_size, self.action_dim, self.agent_history_length)
        self.target_q = build_model(self.frame_size, self.action_dim, self.agent_history_length)
        
        # Complie
        self.optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, momentum=self.gradient_momentum, clipnorm=10.)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.q.summary()

    def get_action(self, sequence):
        if self.epsilon >= np.random.rand():
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q(sequence)[0])

    def model_train(self, predicts, targets):
        model_params = self.q.trainable_variables
        # epsilon decay
        if self.epsilon > self.final_exploration:
            self.epsilon -= (self.initial_exploration + self.final_exploration)/self.final_exploration_frame
        # update parameters
        with tf.GradientTape() as g:
            loss = self.loss(targets, predicts)
        g_theta = g.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(g_theta, model_params))

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
            initial_frame = self.env.reset()
            seq = [preprocess(initial_frame, crop=(0,188,23,136), frame_size=self.frame_size)]
            for _ in range(self.agent_history_length-1):
                obs, _, _, _ = self.env.step(0)
                seq.append(preprocess(obs, crop=(0,188,23,136), frame_size=self.frame_size))
            seq = np.stack(seq, axis=3)
            seq = np.reshape(seq, (1, self.frame_size, self.frame_size, self.agent_history_length))

            while not done:

                frames += 1
                
                # render
                # self.env.render()

                # get action
                action = self.get_action(seq)

                if frames % self.skip_frames != 0:
                    _, _, _, _ = self.env.step(keep_action)
                    continue
                keep_action = action

                # observe next frame
                observation, reward, done, info = self.env.step(action)
                # modify reward
                reward = np.clip(reward, -1, 1)
                # preprocess for next sequence
                next_seq = np.append(preprocess(observation), seq[..., :3], axis=3)
                # store transition in replay memory
                self.replay_memory.append(seq, action, reward, next_seq, done)

                # wait for fill data in replay memory
                if self.replay_memory.crt_idx < self.replay_start_size and not self.replay_memory.is_full():
                    seq = next_seq
                    continue

                """
                Train Q
                """
                # sample batch
                seqs, actions, rewards, next_seqs, dones = self.replay_memory.sample(self.batch_size)

                # argmax action from current q
                a_next_action = self.q(next_seqs)
                argmax_action = np.argmax(a_next_action, axis=1)
                argmax_action = tf.one_hot(argmax_action, self.action_dim)

                # calculate Q(s', a')
                target_qs = self.target_q(next_seqs)

                # Double dqn
                targets = rewards + (1 - dones) * (self.discount_factor * tf.reduce_sum(target_qs * argmax_action, axis=1))

                # seqs for train
                train_seqs = np.reshape(seqs, (self.batch_size, self.frame_size, self.frame_size, self.agent_history_length))
                train_actions = tf.one_hot(actions, self.action_dim)
                predicts = tf.reduce_sum(train_seqs * train_actions, axis=1)

                self.model_train(predicts, targets)

                seq = next_seq
                
                # For logs
                q = self.q(seq)
                if max_q_value < np.amax(q):
                    max_q_value = np.amax(q)
                sum_q_value += np.mean(q)

                # total reward
                episode_reward += reward

                if done:
                    train_ep += 1
                    mean_q_value = sum_q_value / frames * 4
                    if train_ep % self.target_network_update_frequency == 0:
                        self.target_q.model.set_weights(self.q.model.get_weights())
                    crt_buffer_idx = self.replay_memory.crt_idx
                    if self.replay_memory.is_full():
                        crt_buffer_idx = 'full'
                    print('episode: {}, Reward: {}, Epsilon: {:.5f}, buffer size: {}, max Q-value: {:.3f} Q-value: {:.2f}'.format(train_ep,
                                                                                                                            episode_reward,
                                                                                                                            self.q.epsilon,
                                                                                                                            crt_buffer_idx,
                                                                                                                            max_q_value,
                                                                                                                            mean_q_value))
                    wandb.log({'Reward':episode_reward, 'Q value':mean_q_value})

            if train_ep % 500 == 0:
                self.q.save_weights('/home/ubuntu/RL_Practices/model_free/atari_dqn_ram/save_weights/dqn_pong_' + str(train_ep) + 'epi.h5')