from replay_memory import ReplayMemory
from dqn_neural_net import DQN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque



class DQNAgent(object):

    def __init__(self, env, is_test=False):

        # hyperparameter
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.replay_memory_size = 100000
        self.replay_start_size = 10000
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

        # replay memory with class
        # self.replay_memory = ReplayMemory(self.replay_memory_size, self.state_dim)
        # replay memory with deque
        self.replay_memory = deque(maxlen=self.replay_memory_size)


        # Q function
        self.q = DQN(self.state_dim, self.action_dim, self.LEARNING_RATE)
        self.target_q = DQN(self.state_dim, self.action_dim, self.LEARNING_RATE)

        # total reward of a episode
        self.save_epi_reward = []

        self.save_mean_q_value = []

        self.stop_train = 4

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
            # targets = np.empty([self.BATCH_SIZE], dtype=np.float32)
            targets = deque(maxlen=self.BATCH_SIZE)

            # init position of car at last frame
            end_position = self.env.min_position

            states = deque(maxlen=self.BATCH_SIZE)
            actions = deque(maxlen=self.BATCH_SIZE)
            rewards = deque(maxlen=self.BATCH_SIZE)
            next_states = deque(maxlen=self.BATCH_SIZE)
            dones = deque(maxlen=self.BATCH_SIZE)

            while not done:

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

                # # sample random mini batch of transitions from replay memory
                # states, actions, rewards, next_states, dones = self.replay_memory.sample(self.BATCH_SIZE)

                # deque reply memory
                random_index = np.random.choice(len(self.replay_memory), 1)[0]

                states.append((self.replay_memory[random_index][0]))
                actions.append((self.replay_memory[random_index][1]))
                rewards.append(self.replay_memory[random_index][2])
                next_states.append(self.replay_memory[random_index][3])
                dones.append(self.replay_memory[random_index][4])

                target = dones[-1] * rewards[-1] \
                         + (1 - dones[-1]) * (rewards[-1] + self.discount_factor
                                              * np.amax(self.target_q.model(next_states[-1])))
                targets.append(target)

                if len(targets) == self.BATCH_SIZE:
                    # next target q value and q value with action
                    # next_target_q_value = self.target_q.model(next_states)
                    # next_q_value = self.q.model(next_states)

                    # calculate target
                    # for i in range(self.BATCH_SIZE):
                    #     # # DQN
                    #     targets[i] = dones[i]*rewards[i]+(1-dones[i])*(rewards[i]+self.discount_factor * np.amax(next_target_q_value[i]))
                    #
                    #     # # DDQN
                    #     # argmx_action = self.q.get_action(np.reshape(states[i], (1, self.state_dim)))
                    #     # targets[i] = dones[i] * rewards[i] + (1-dones[i]) * (self.discount_factor * next_target_q_value[i][argmx_action])

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
                    Q_value = self.q.model(state)
                    print('Episode: {}, Reward: {}, End Position: {:.3f}, Epsilon: {:.5f}, Q-value: {}'.format(train_ep, episode_reward, end_position, self.q.initial_exploration, np.mean(Q_value)))
                    self.save_epi_reward.append(episode_reward)
                    self.save_mean_q_value.append(np.mean(Q_value[0]))

            if e % self.target_network_update_frequency == 0:
                self.q.save_weights('./save_weights/mountainCar_dqn.h5')
                # self.target_q.load_weights('./save_weights/')
                self.target_q.model.set_weights(self.q.model.get_weights())

            # stop train condition
            if end_position >= self.max_position:
                stop_train_count += 1
            else:
                stop_train_count = 0

    # graph episodes and rewards
    def plot_result(self):
        plt.subplot(211)
        plt.plot(self.save_epi_reward)

        plt.subplot(212)
        plt.plot(self.save_mean_q_value)

        plt.show()
