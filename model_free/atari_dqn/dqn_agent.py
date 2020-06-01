from replay_memory import ReplayMemory
from dqn_neural_net import DQN
import numpy as np
import cv2
import tensorflow as tf

class Agent(object):

    def __init__(self, env, is_test=False):

        # hyperparameter
        self.frame_size = 84
        self.batch_size = 32
        self.discount_factor = 0.99
        self.replay_start_size = 5000
        self.target_network_update_frequency = 5
        self.agent_history_length = 4
        self.replay_memory_size = 10000
        self.batch_shape = (self.batch_size, self.frame_size, self.frame_size, self.agent_history_length)
        self.action_repeat = 4
        self.update_frequency = 4

        # test environment
        if is_test:
            self.replay_memory_size = 32

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


    def preprocess(self, frame):
        frame = np.reshape(cv2.resize(frame, dsize=(self.frame_size, self.frame_size))[..., 0], (1, self.frame_size, self.frame_size, 1))
        return np.array(frame, dtype=np.float32) / 255

    def train(self, max_episode_num):

        # repeat episode
        for e in range(int(max_episode_num)):

            # init batch
            targets = []

            # initialize sequence
            initial_frame = self.env.reset()
            seq = [self.preprocess(initial_frame)]
            for _ in range(self.agent_history_length-1):
                obs, _, _, _ = self.env.step(0)
                seq.append(self.preprocess(obs))
            seq = np.stack(seq, axis=3).reshape(1, self.frame_size, self.frame_size, self.agent_history_length)
            next_seq = seq

            # init done, total reward, frames, action
            done = False
            total_reward = 0
            frames = 0
            action = 0


            while not done:

                # # render
                # self.env.render()

                # repeat action
                if frames % self.action_repeat == 0:
                    # get action
                    action = self.q.get_action(seq)

                    # observe next frame
                    observation, reward, done, info = self.env.step(action)

                    # preprocess for next sequence
                    next_seq = np.append(self.preprocess(observation), seq[..., :3], axis=3)

                    # store transition in replay memory
                    self.replay_memory.append(seq, action, reward, next_seq, done)

                _, reward , done, _ = self.env.step(action)

                # total reward
                total_reward += reward

                # add frame
                frames += 1

                # wait for full replay memory
                if self.replay_memory.current < self.replay_start_size or self.replay_memory.is_full:
                    seq = next_seq
                    continue

                # train for each 4 frames
                if frames % self.update_frequency * self.action_repeat == 0:

                    # sample random mini batch of transitions from replay memory
                    seqs, actions, rewards, next_seqs, dones = self.replay_memory.sample(self.batch_size)

                    # next target q value and q value with action
                    next_target_q_value = self.target_q.model(next_seqs)
                    next_q_value = self.q.model(next_seqs)

                    # calculate target
                    for i in range(self.batch_size):
                        if dones[i]:
                            targets.append(rewards[i])
                        else:
                            # DQN
                            targets.append(rewards[i] + self.discount_factor * np.amax(next_target_q_value[i]))
                            # DDQN
                            # targets.append(rewards[i] + self.discount_factor * tf.gather(next_target_q_value[i], [np.argmax(next_q_value[i])]))
                    # train
                    self.q.train(targets, seqs, actions)

                seq = next_seq
                targets = []

            print("Episode: ", e + 1, "total reward: ", total_reward, "epsilon: ", self.q.epsilon)
            if e > 100:
                print('mean q value: ', np.mean(self.q.model(seq)[0]))

            if e % self.target_network_update_frequency == 0:
                self.q.save_weights('./save_weights/dqn.h5')
                self.target_q.load_weights('./save_weights/')

    def test(self):

        # init batch
        targets = []

        # initialize sequence
        initial_frame = self.env.reset()
        seq = [self.preprocess(initial_frame)]
        for _ in range(self.agent_history_length - 1):
            obs, _, _, _ = self.env.step(0)
            seq.append(self.preprocess(obs))
        seq = np.stack(seq, axis=3).reshape(1, self.frame_size, self.frame_size, self.agent_history_length)
        next_seq = seq

        # init done, total reward, frames, action
        done = False
        total_reward = 0
        frames = 0
        action = 0

        while not done:

            # get action
            action = self.q.get_action(seq)

            # observe next frame
            observation, reward, done, info = self.env.step(action)

            # preprocess for next sequence
            next_seq = np.append(self.preprocess(observation), seq[..., :3], axis=3)

            # store transition in replay memory
            self.replay_memory.append(seq, action, reward, next_seq, done)

            _, reward, done, _ = self.env.step(action)

            # total reward
            total_reward += reward

            # add frame
            frames += 1

            # sample random mini batch of transitions from replay memory
            seqs, actions, rewards, next_seqs, dones = self.replay_memory.sample(self.batch_size)

            # next target q value and q value with action
            next_target_q_value = self.target_q.model(next_seqs)

            # calculate target
            for i in range(self.batch_size):
                if dones[i]:
                    targets.append(rewards[i])
                else:
                    targets.append(rewards[i] + self.discount_factor * np.amax(next_target_q_value[i]))

            # train
            self.q.train(targets, seqs, actions)
            break

        self.q.load_weights('./save_weights/')

        done = False
        initial_frame = self.env.reset()
        seq = [self.preprocess(initial_frame)]
        for _ in range(self.agent_history_length - 1):
            obs, _, _, _ = self.env.step(0)
            seq.append(self.preprocess(obs))
        seq = np.stack(seq, axis=3).reshape(1, self.frame_size, self.frame_size, self.agent_history_length)

        while not done:

            self.env.render()
            # get action
            action = self.q.get_action(seq, is_test=True)

            # observe next frame
            observation, _, done, _ = self.env.step(action)

            # preprocess for next sequence
            seq = np.append(self.preprocess(observation), seq[..., :3], axis=3)
