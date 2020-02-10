import numpy as np
import random

class ReplayMemory:

    def __init__(self, config):

        self.config = config
        self.current = 0
        self.current_memory_size = 0

        self.memory_size = self.config.memory_size
        self.batch_size = self.config.batch_size

        self.states = np.empty((self.memory_size,self.config.frame_height, self.config.frame_width, 4), dtype=np.float16)
        self.actions = np.empty(self.memory_size, np.uint8)
        self.rewards = np.empty(self.memory_size, np.float16)
        self.poststates = np.empty((self.memory_size, self.config.frame_height, self.config.frame_width, 4), dtype=np.float16)
        self.dones = np.empty(self.memory_size, np.bool)

        self.indices = []


    def add(self, state, action, reward, poststate, done):

        self.states[self.current, ...] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.poststates[self.current, ...] = poststate
        self.dones[self.current] = done

        self.current = (self.current + 1) % self.memory_size
        self.current_memory_size += 1

    def sample(self):

        self.indices = []

        if self.current_memory_size < self.memory_size:
            potential_batch_size = self.current_memory_size
        else:
            potential_batch_size = self.memory_size

        for _ in range(self.batch_size):
            self.indices.append(random.randrange(potential_batch_size))
        
        return self.states[self.indices,...], self.actions[self.indices], self.rewards[self.indices], \
               self.poststates[self.indices], self.dones[self.indices]

