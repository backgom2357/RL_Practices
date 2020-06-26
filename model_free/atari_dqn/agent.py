from neural_net import DQN
from replay_memory import ReplayMemory
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time

class Agent(object):

    def __init__(self, env):

        # hyperparameter
        self.frame_size = 48
        self.batch_size = 32
        self.discount_factor = 0.99
        self.target_network_update_frequency = 5
        self.agent_history_length = 4
        self.replay_memory_size = 100000
        self.replay_start_size = 50000
        self.action_repeat = 4
        self.update_frequency = 4

        # environment
        self.env = env
        # state dimension
        self.state_dim = env.observation_space.shape[0]
        # action dimension
        self.action_dim = env.action_space.n
        # replay memory
        self.replay_memory = ReplayMemory(self.replay_memory_size, self.frame_size, self.agent_history_length)

        # Q function
        self.q = DQN(self.action_dim)
        self.target_q = DQN(self.action_dim)

        # total reward of a episode
        self.save_epi_reward = []
        self.save_mean_q_value = []

        self.epi_per_epoch = 42

        # self.stop_train = 30

    def preprocess(self, frame):
        frame = np.reshape(cv2.resize(frame[0:188, 23:136, :], dsize=(self.frame_size, self.frame_size))[..., 0],
                           (1, self.frame_size, self.frame_size, 1))
        return np.array(frame, dtype=np.float32) / 255

    def train(self, epoch):

        train_ep = 0
        # stop_train_count = 0
        self.replay_memory.reset()

        for epoch in range(epoch):

            mean_q_value_per_epoch = 0
            episode_reward_per_epoch = 0

            # repeat episode
            for e in range(int(self.epi_per_epoch)):

                # if stop_train_count > self.stop_train:
                #     self.q.save_weights('./save_weights/boxing_dqn.h5')
                #     print("이제 잘하네!")
                #     break

                # initialize frames, episode_reward, done
                frames, done = 0, False
                sum_q_value = 0

                # reset env and observe initial state
                initial_frame = self.env.reset()
                seq = [self.preprocess(initial_frame)]
                for _ in range(self.agent_history_length-1):
                    obs, _, _, _ = self.env.step(0)
                    seq.append(self.preprocess(obs))
                seq = np.stack(seq, axis=3)
                seq = np.reshape(seq, (1, self.frame_size, self.frame_size, self.agent_history_length))

                while not done:

                    frames += 1
                    # render
                    self.env.render()
                    # get action
                    action = self.q.get_action(seq)
                    # observe next frame
                    observation, reward, done, info = self.env.step(action)
                    # modify reward
                    reward = np.clip(reward, -1, 1)
                    # preprocess for next sequence
                    next_seq = np.append(self.preprocess(observation), seq[..., :3], axis=3)
                    # store transition in replay memory
                    self.replay_memory.append(seq, action, reward, next_seq, done)

                    # wait for full replay memory
                    if self.replay_memory.crt_idx < self.replay_start_size or self.replay_memory.is_full:
                        seq = next_seq
                        continue

                    # sample batch
                    seqs, actions, rewards, next_seqs, dones = self.replay_memory.sample(self.batch_size)

                    # argmax action from current q
                    a_next_action = self.q.model(next_seqs)[1]
                    argmax_action = np.argmax(a_next_action, axis=1)
                    argmax_action = tf.one_hot(argmax_action, self.action_dim)

                    # calculate Q(s', a')
                    target_vs, target_as = self.target_q.model(next_seqs)
                    target_qs = target_as \
                                + (target_vs - tf.reshape(tf.reduce_mean(target_as, axis=1), shape=(len(target_as), 1)))

                    # Double dqn
                    targets = rewards + (1 - dones) * (self.discount_factor * tf.reduce_sum(target_qs * argmax_action, axis=1))

                    # train
                    input_states = np.reshape(seqs, (self.batch_size, self.frame_size, self.frame_size, self.agent_history_length))
                    input_actions = tf.one_hot(actions, self.action_dim)
                    self.q.train(input_states, input_actions, targets)

                    seq = next_seq

                    v, a = self.q.model(seq)
                    q = v + (a - tf.reduce_mean(a))
                    sum_q_value += np.mean(q)

                    # total reward
                    episode_reward_per_epoch += reward

                train_ep += 1
                mean_q_value_per_epoch = sum_q_value / frames
                if train_ep % self.target_network_update_frequency == 0:
                    self.target_q.model.set_weights(self.q.model.get_weights())

            print('Epoch: {}, Reward: {}, Epsilon: {:.5f}, Q-value: {}'.format(epoch,
                                                                                 episode_reward_per_epoch,
                                                                                 self.q.epsilon,
                                                                                 mean_q_value_per_epoch))
            self.save_epi_reward.append(episode_reward_per_epoch)
            self.save_mean_q_value.append(mean_q_value_per_epoch)



            if epoch % 10 == 0:
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

    # graph episodes and rewards
    def plot_result(self):
        plt.subplot(211)
        plt.plot(self.save_epi_reward)

        plt.subplot(212)
        plt.plot(self.save_mean_q_value)

        plt.savefig('reward_meanQ.png')

        plt.show()