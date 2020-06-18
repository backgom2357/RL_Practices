import numpy as np

class ReplayMemory(object):
    def __init__(self, replay_memory_size, state_dim, action_dim):
        self.rm_size = replay_memory_size

        # init state, action, reward, next_state, done
        self.states = np.zeros((self.rm_size, state_dim))
        self.actions = np.zeros((self.rm_size,))
        self.rewards = np.zeros((self.rm_size,))
        self.next_states = np.zeros((self.rm_size, state_dim))
        self.dones = np.zeros((self.rm_size,))

        self.crt_idx = 0
        self.is_full = self.rewards[-1] != 0 # hope the last reward is not 0

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        self.crt_idx = (self.crt_idx + 1) % self.rm_size

    def sample(self, batch_size):
        rd_idx = np.random.choice((1 - self.is_full)*self.crt_idx+self.is_full*self.rm_size, batch_size)
        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones