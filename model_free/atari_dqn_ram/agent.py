from neural_net import DQN
from replay_memory_ram import ReplayMemory
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

class Agent(object):

    def __init__(self, env):

        # hyperparameter
        self.frame_size = 84 # 크기를 키우면 allocate memory problem이 난다.
        self.batch_size = 64
        self.discount_factor = 0.99
        self.target_network_update_frequency = 5
        self.agent_history_length = 4
        self.update_frequency = 4
        self.skip_frames = 4

        # environment
        self.env = env
        # state dimension
        self.state_dim = env.observation_space.shape[0]
        # action dimension
        # self.action_dim = env.action_space.n
        self.action_dim = 5
        # replay memory
        self.replay_memory_size = 600000
        self.replay_start_size = 300000
        self.replay_memory = ReplayMemory(self.replay_memory_size, self.frame_size, self.agent_history_length)

        # Q function
        self.q = DQN(self.frame_size, self.action_dim, self.agent_history_length)
        self.target_q = DQN(self.frame_size, self.action_dim, self.agent_history_length)

        # total reward of a episode
        self.save_epi_reward = []
        self.save_mean_q_value = []

        # self.stop_train = 30

    def preprocess(self, frame):
        frame = np.reshape(cv2.resize(frame[0:188, 23:136, :], dsize=(self.frame_size, self.frame_size))[..., 0],
                           (1, self.frame_size, self.frame_size, 1))
        return np.array(frame, dtype=np.float32) / 255

    def train(self, episodes):

        train_ep = 0

        # repeat episode
        for e in range(episodes):

            # if stop_train_count > self.stop_train:
            #     self.q.save_weights('./save_weights/boxing_dqn.h5')
            #     print("이제 잘하네!")
            #     break

            # initialize frames, episode_reward, done
            frames, done = 0, False
            sum_q_value = 0
            episode_reward = 0
            keep_action = 0
            max_q_value = -999

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
                # self.env.render()
                # get action
                action = self.q.get_action(seq)

                if frames % self.skip_frames != 0:
                    _, _, _, _ = self.env.step(keep_action)
                    continue
                keep_action = action

                # observe next frame
                observation, reward, done, info = self.env.step(action)
                # modify reward
                reward = np.clip(reward, -1, 1)
                # preprocess for next sequence
                next_seq = np.append(self.preprocess(observation), seq[..., :3], axis=3)
                # store transition in replay memory
                self.replay_memory.append(seq, action, reward, next_seq, done)

                # wait for fill data in replay memory
                if self.replay_memory.crt_idx < self.replay_start_size and not self.replay_memory.is_full():
                    seq = next_seq
                    continue

                # sample batch
                seqs, actions, rewards, next_seqs, dones = self.replay_memory.sample(self.batch_size)

                # argmax action from current q
                a_next_action = self.q.model(next_seqs)
                argmax_action = np.argmax(a_next_action, axis=1)
                argmax_action = tf.one_hot(argmax_action, self.action_dim)

                # calculate Q(s', a')
                target_qs = self.target_q.model(next_seqs)

                # Double dqn
                targets = rewards + (1 - dones) * (self.discount_factor * tf.reduce_sum(target_qs * argmax_action, axis=1))

                # train
                input_states = np.reshape(seqs, (self.batch_size, self.frame_size, self.frame_size, self.agent_history_length))
                input_actions = tf.one_hot(actions, self.action_dim)

                self.q.train(input_states, input_actions, targets)

                seq = next_seq

                q = self.q.model(seq)
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
                    self.save_epi_reward.append(episode_reward)
                    self.save_mean_q_value.append(mean_q_value)

            if train_ep % 100 == 0:
                self.q.save_weights('/home/ubuntu/RL_Practices/model_free/atari_dqn_ram/save_weights/dqn_boxing_' + str(train_ep) + 'epi.h5')

        np.savetxt('/home/ubuntu/RL_Practices/model_free/atari_dqn_ram/save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        np.savetxt('/home/ubuntu/RL_Practices/model_free/atari_dqn_ram/save_weights/pendulum_epi_reward.txt', self.save_mean_q_value)


    def test(self, path):

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
        self.q.model.load_weights(path)

        while not done:

            time.sleep(0.05)

            frames += 1
            # # render
            # self.env.render()
            # get action
            action = np.argmax(self.q.model(seq)[0])
            # observe next frame
            observation, reward, done, info = self.env.step(action)
            # preprocess for next sequence
            next_seq = np.append(self.preprocess(observation), seq[..., :3], axis=3)
            # store transition in replay memory
            seq = next_seq

            # check what the agent see
            test_img = np.reshape(next_seq, (84, 84, 4))
            test_img = cv2.resize(test_img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
            cv2.imshow('obs', test_img)
            if cv2.waitKey(25)==ord('q') or done:
                cv2.destroyAllWindows()

            print(action, self.q.model(seq)[1], end='\r')

    # graph episodes and rewards
    def plot_result(self):
        plt.subplot(211)
        plt.plot(self.save_epi_reward)

        plt.subplot(212)
        plt.plot(self.save_mean_q_value)

        plt.savefig('reward_meanQ.png')

        plt.show()