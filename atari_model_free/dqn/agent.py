import tensorflow as tf
from .NeuralQLearner.dqn import QValue
from .replay_memory import ReplayMemory
from .util import Util
import numpy as np
from tqdm import tqdm

class Agent:

    def __init__(self, config, env):

        self.env = env
        self.action_space = env.action_space.n

        self.config = config
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.batch_size = config.batch_size
        self.discount_factor = config.discount_factor

        self.replay_memory = ReplayMemory(self.config)

        self.model = QValue(self.config).DQN(self.action_space)
        self.target_model = QValue(self.config).DQN(self.action_space)

        self.util = Util(self.config)

        self.average_loss = 0

    def train_model(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        states, actions, rewards, poststates, dones = self.replay_memory.sample()

        target = self.model.predict(states)
        target_opt = self.target_model.predict(poststates)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_opt[i])

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, is_train):
        if self.epsilon >= np.random.rand() and is_train:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.target_model.predict(state)[0])

    def play(self):

        score = 0

        obs = self.util.preprocess(self.env.reset())

        state = np.stack((obs, obs, obs, obs), axis=3).reshape(
            (1, self.config.frame_height, self.config.frame_width, 4))

        for s in tqdm(range(self.config.steps)):

            action = self.get_action(state, True)

            next_obs, reward, done, info = self.env.step(action)

            reward = np.clip(reward, -1, 1)
            score += reward

            poststate = np.append(self.util.preprocess(next_obs), state[..., :3], axis=3)
            self.replay_memory.add(state, action, reward, poststate, done)

            if self.config.start_train <= s:
                if s % self.config.train_frequency == self.config.train_frequency - 1:
                    self.train_model()

            state = poststate

            if (s+1) % 10000 == 0:
                self.update_target_model()
                self.model.save_weights("./boxing-dqn.h5")

            if done:
                score = 0
                obs = self.util.preprocess(self.env.reset())

                state = np.stack((obs, obs, obs, obs), axis=3).reshape(
                    (1, self.config.frame_height, self.config.frame_width, 4))

            if s % 50000 == 0 and s > 50000:
                self.evaluation(self.config.render)

    def evaluation(self, render):

        done = False
        score = 0
        obs = self.util.preprocess(self.env.reset())
        state = np.stack((obs, obs, obs, obs), axis=3).reshape(
            (1, self.config.frame_height, self.config.frame_width, 4))

        while not done:

            if render:
                self.env.render()

            action = self.get_action(state, False)
            next_obs, reward, done, info = self.env.step(action)
            reward = np.clip(reward, -1, 1)
            score += reward
            state = np.append(self.util.preprocess(next_obs), state[..., :3], axis=3)

        print('average score : %d, average_q : %d' % score)
