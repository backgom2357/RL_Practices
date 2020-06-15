from dqn_neural_net import DQN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import time



class DQNAgent(object):

    def __init__(self, env, is_test=False):

        # hyperparameter
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.replay_memory_size = 40000
        self.replay_start_size = 5000
        self.discount_factor = 0.99
        self.target_network_update_frequency = 5

        # environment
        self.env = env
        # state dimension
        self.state_dim = env.observation_space.shape[0]
        # action dimension
        self.action_dim = self.env.action_space.n
        # max position
        self.max_position = -1.2
        # replay memory with deque
        self.replay_memory = deque(maxlen=self.replay_memory_size)


        # Q function
        self.q = DQN(self.state_dim, self.action_dim, self.LEARNING_RATE)
        self.target_q = DQN(self.state_dim, self.action_dim, self.LEARNING_RATE)

        # total reward of a episode
        self.save_epi_reward = []
        self.save_mean_q_value = []

        self.stop_train = 10
    def train(self, max_episode_num):

        train_ep = 0
        stop_train_count = 0

        # repeat episode
        for e in range(int(max_episode_num)):
        # for e in range(1):

            # stop train
            if stop_train_count > self.stop_train:
                self.q.save_weights('./save_weights/mountainCar_dqn.h5')
                print("이제 잘하네!")
                break

            # init episode
            time, episode_reward, done = 0, 0, False
            # reset env and observe initial state
            state = self.env.reset()
            state = np.reshape(state, (1, self.state_dim))

            targets = deque(maxlen=self.BATCH_SIZE)
            states = deque(maxlen=self.BATCH_SIZE)
            actions = deque(maxlen=self.BATCH_SIZE)
            rewards = deque(maxlen=self.BATCH_SIZE)
            next_states = deque(maxlen=self.BATCH_SIZE)
            dones = deque(maxlen=self.BATCH_SIZE)

            mean_q_value = 0
            frame = 0

            while not done:

                frame += 1

                # render
                self.env.render()

                # get action
                action = self.q.get_action(state)

                # observe next state, reward
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, self.state_dim))

                # Modify reward
                if next_state[0, 0] > self.max_position:
                    self.max_position = next_state[0, 0]
                    reward = 10
                if next_state[0, 0] >= 0.5:
                    reward += 10

                # reshape
                self.replay_memory.append((state, action, reward, next_state, done))

                # wait for full replay memory
                if len(self.replay_memory) < self.replay_start_size:
                    state = next_state
                    continue

                # append 1 random sample
                random_index = np.random.choice(len(self.replay_memory), 1)[0]

                states.append((self.replay_memory[random_index][0]))
                actions.append((self.replay_memory[random_index][1]))
                sampled_reward = self.replay_memory[random_index][2]
                sampled_done = self.replay_memory[random_index][4]
                sampled_next_state = self.replay_memory[random_index][3]

                # argmax action from current q
                argmax_action = np.argmax(self.q.model(sampled_next_state)[1])
                argmax_action = tf.one_hot(argmax_action, self.action_dim)

                target_v, target_a = self.target_q.model(sampled_next_state)
                target_q = target_v + (target_a - tf.reduce_mean(target_a))

                # Double dqn
                target = sampled_done * sampled_reward \
                         + (1 - sampled_done) * (sampled_reward + self.discount_factor
                                                 * (np.dot(target_q, argmax_action)))
                targets.append(target)

                v, a = self.q.model(state)
                q = v + (a - tf.reduce_mean(a))
                mean_q_value += np.mean(q)

                if len(targets) == self.BATCH_SIZE:

                    # train
                    input_states = np.reshape(states, (self.BATCH_SIZE, self.state_dim))
                    input_actions = tf.one_hot(actions, self.action_dim)
                    self.q.train_on_batch(input_states, input_actions, targets)

                state = next_state
                episode_reward += reward
                time += 1

                if done:
                    train_ep += 1
                    end_position = state[0, 0]
                    print('Episode: {}, Reward: {}, End Position: {:.3f}, Epsilon: {:.5f}, Q-value: {}'.format(train_ep, episode_reward, end_position, self.q.initial_exploration, mean_q_value/frame))
                    self.save_epi_reward.append(episode_reward)
                    self.save_mean_q_value.append(mean_q_value/frame)

                    if train_ep % self.target_network_update_frequency == 0:
                        self.q.model.save_weights('./save_weights/mountainCar_dqn.h5')
                        self.target_q.model.set_weights(self.q.model.get_weights())

                    # stop train condition
                    if episode_reward > -200:
                        stop_train_count += 1
                    else:
                        stop_train_count = 0

    def test(self):

        train_ep = 0

        self.q.initial_exploration = 0.

        # init episode
        episode_reward, done = 0, False
        # reset env and observe initial state
        state = self.env.reset()
        state = np.reshape(state, (1, self.state_dim))

        targets = deque(maxlen=self.BATCH_SIZE)
        states = state
        actions = [0]

        mean_q_value = 0
        frame = 0
        input_states = np.reshape(states, (1, self.state_dim))
        input_actions = tf.one_hot(actions, self.action_dim)
        self.q.train_on_batch(input_states, input_actions, targets)
        self.q.model.load_weights('./save_weights/mountainCar_dqn.h5')

        while not done:

            frame += 1

            time.sleep(0.003)

            # render
            self.env.render()

            # get action
            action = self.q.get_action(state)

            # observe next state, reward
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, (1, self.state_dim))

            state = next_state
            episode_reward += reward

            if done:
                train_ep += 1
                end_position = state[0, 0]
                print('Episode: {}, Reward: {}, End Position: {:.3f}, Epsilon: {:.5f}, Q-value: {}'.format(train_ep,
                                                                                                           episode_reward,
                                                                                                           end_position,
                                                                                                           self.q.initial_exploration,
                                                                                                           mean_q_value / frame))

    # graph episodes and rewards
    def plot_result(self):
        plt.subplot(211)
        plt.plot(self.save_epi_reward)

        plt.subplot(212)
        plt.plot(self.save_mean_q_value)

        plt.savefig('reward_meanQ.png')

        plt.show()
