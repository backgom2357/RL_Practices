from dqn_neural_net import DQN
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import time

class Agent(object):

    def __init__(self, env):

        # hyperparameter
        self.frame_size = 84
        self.batch_size = 32
        self.discount_factor = 0.99
        self.target_network_update_frequency = 5
        self.agent_history_length = 4
        self.replay_memory_size = 80000
        self.replay_start_size = 40000
        self.action_repeat = 4
        self.update_frequency = 4

        # environment
        self.env = env
        # state dimension
        self.state_dim = env.observation_space.shape[0]
        # action dimension
        self.action_dim = env.action_space.n

        # replay memory
        self.replay_memory = deque(maxlen=self.replay_memory_size)

        # Q function
        self.q = DQN(self.action_dim)
        self.target_q = DQN(self.action_dim)

        # total reward of a episode
        self.save_epi_reward = []
        self.save_mean_q_value = []


    def preprocess(self, frame):
        frame = np.reshape(cv2.resize(frame, dsize=(self.frame_size, self.frame_size))[..., 0],
                           (1, self.frame_size, self.frame_size, 1))
        return np.array(frame, dtype=np.float32) / 255

    def train(self, max_episode_num):

        train_ep = 0

        # repeat episode
        for e in range(int(max_episode_num)):

            # initialize frames, episode_reward, done
            frames, episode_reward, done = 0, 0, False

            # reset env and observe initial state
            initial_frame = self.env.reset()
            seq = [self.preprocess(initial_frame)]
            for _ in range(self.agent_history_length-1):
                obs, _, _, _ = self.env.step(0)
                seq.append(self.preprocess(obs))
            seq = np.stack(seq, axis=3)
            seq = np.reshape(seq, (1, self.frame_size, self.frame_size, self.agent_history_length))

            # init batches
            targets = deque(maxlen=self.batch_size)
            seqs = deque(maxlen=self.batch_size)
            actions = deque(maxlen=self.batch_size)

            # init mean_q_value for plot
            mean_q_value = 0

            while not done:

                frames += 1
                # render
                # self.env.render()
                # get action
                action = self.q.get_action(seq)
                # observe next frame
                observation, reward, done, info = self.env.step(action)
                # modify reward
                reward = np.clip(reward, -1, 1)
                # preprocess for next sequence
                next_seq = np.append(self.preprocess(observation), seq[..., :3], axis=3)
                # store transition in replay memory
                self.replay_memory.append((seq, action, reward, next_seq, done))
                # total reward
                episode_reward += reward

                # wait for full replay memory
                if len(self.replay_memory) < self.replay_start_size:
                    seq = next_seq
                    continue

                # deque reply memory
                random_index = np.random.choice(len(self.replay_memory), 1)[0]

                seqs.append((self.replay_memory[random_index][0]))
                actions.append((self.replay_memory[random_index][1]))
                sampled_reward = self.replay_memory[random_index][2]
                sampled_done = self.replay_memory[random_index][4]
                sampled_next_seq = self.replay_memory[random_index][3]

                # argmax action from current q
                amax_action = np.amax(self.q.model(sampled_next_seq))
                amax_action = tf.one_hot(amax_action, self.action_dim)

                target = sampled_done * sampled_reward \
                         + (1 - sampled_done) * (sampled_reward + self.discount_factor
                                              * (np.dot(self.target_q.model(sampled_next_seq), amax_action)))
                targets.append(target)

                mean_q_value += np.mean(self.q.model(seq))

                # train for each 4 frames
                if len(targets) == self.batch_size:
                    input_seqs = np.reshape(seqs,
                                            (self.batch_size, self.frame_size, self.frame_size,
                                             self.agent_history_length))
                    input_actions = tf.one_hot(actions, self.action_dim)
                    self.q.train(input_seqs, input_actions, targets)

                seq = next_seq

                if done:
                    train_ep += 1

                    self.save_epi_reward.append(episode_reward)
                    self.save_mean_q_value.append(mean_q_value / frames)
                    if train_ep % 40 == 0:
                        self.test(path="./save_weights/", is_valid=True)
                        print('Episode: {}, Reward: {}, Epsilon: {:.5f}, Q-value: {}'.format(train_ep,
                                                                                         episode_reward,
                                                                                         self.q.epsilon,
                                                                                         mean_q_value / frames))
                    if train_ep % self.target_network_update_frequency == 0:
                        self.target_q.model.set_weights(self.q.model.get_weights())

                    if train_ep % 100 == 0:
                        self.q.save_weights('./save_weights/dqn_boxing_' + str(train_ep) + 'epi.h5')

        np.savetxt('.save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        np.savetxt('.save_weights/pendulum_epi_reward.txt', self.save_mean_q_value)


    def test(self, path, is_valid=False):

        train_ep = 0

        # initialize sequence
        episode_reward, done = 0, False
        # reset env and observe initial state
        initial_frame = self.env.reset()
        seq = [self.preprocess(initial_frame)]
        for _ in range(self.agent_history_length - 1):
            obs, _, _, _ = self.env.step(0)
            seq.append(self.preprocess(obs))
        seq = np.stack(seq, axis=3)
        seq = np.reshape(seq, (1, self.frame_size, self.frame_size, self.agent_history_length))

        # init done, total reward, frames, action
        frames = 0
        mean_q_value = 0
        self.q.train(seq, 0, 0)
        if not is_valid:
            self.q.model.load_weights(path)

        while not done:

            time.sleep(0.01)

            frames += 1
            # # render
            self.env.render()
            # get action
            action = np.argmax(self.q.model(seq)[0])
            # observe next frame
            observation, reward, done, info = self.env.step(action)
            # preprocess for next sequence
            next_seq = np.append(self.preprocess(observation), seq[..., :3], axis=3)
            # store transition in replay memory
            seq = next_seq

            if done:
                train_ep += 1
                print('Episode: {}, Reward: {}, Epsilon: {:.5f}, Q-value: {}'.format(train_ep,
                                                                                     episode_reward,
                                                                                     self.q.epsilon,
                                                                                     mean_q_value / frames))

    # graph episodes and rewards
    def plot_result(self):
        plt.subplot(211)
        plt.plot(self.save_epi_reward)

        plt.subplot(212)
        plt.plot(self.save_mean_q_value)

        plt.savefig('reward_meanQ.png')

        plt.show()