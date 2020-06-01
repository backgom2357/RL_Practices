import numpy as np

class ReplayMemory(object):

    def __init__(self, replay_memory_size, state_dim):

        self.current = 0
        self.is_full = False
        self.replay_memory_size = replay_memory_size

        self.seqs = np.empty((replay_memory_size, state_dim), dtype=np.float32)
        self.actions = np.empty(replay_memory_size)
        self.rewards = np.empty(replay_memory_size)
        self.next_seqs = np.empty((replay_memory_size, state_dim), dtype=np.float32)
        self.dones = np.empty(replay_memory_size, dtype=np.bool)

    def append(self, seq, action, reward, next_seq, done):

        self.seqs[self.current, ...] = seq
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.next_seqs[self.current, ...] = next_seq
        self.dones[self.current] = done

        self.current = (self.current + 1) % self.replay_memory_size
        if self.current == self.replay_memory_size:
            self.is_full = True

    def sample(self, batch_size):

        if self.is_full:
            indices = np.random.choice(self.replay_memory_size, batch_size)
        else:
            indices = np.random.choice(self.current, batch_size)

        return self.seqs[indices, ...], self.actions[indices], self.rewards[indices], self.next_seqs[indices, ...], self.dones[indices]