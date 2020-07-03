import numpy as np
import os

class ReplayMemory(object):
    def __init__(self, replay_memory_size, frame_size, agent_history_length, max_files_num):
        self.replay_memory_size = replay_memory_size
        self.frame_size = frame_size
        self.agent_history_length = agent_history_length
        self.max_files_num = max_files_num
        self.file_idx = 0

        # init seqs, action, reward, next_seqs, done
        self.seqs = np.zeros((replay_memory_size, frame_size, frame_size, agent_history_length), dtype=np.float32)
        self.actions = np.zeros(replay_memory_size, np.uint8)
        self.rewards = np.zeros(replay_memory_size, np.float32)
        self.next_seqs = np.zeros((replay_memory_size, frame_size, frame_size, agent_history_length), dtype=np.float32)
        self.dones = np.zeros(replay_memory_size, np.bool)

        self.crt_idx = 0
        self.is_full = False

    def append(self, seq, action, reward, next_seq, done):
        self.seqs[self.crt_idx, ...] = seq
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_seqs[self.crt_idx, ...] = next_seq
        self.dones[self.crt_idx] = done

        self.crt_idx += 1

        if self.crt_idx == self.replay_memory_size:
            self.is_full = True
            np.savez('./replay_data/'+str(self.file_idx)+'.npz',  
            seqs=self.seqs, 
            actions=self.actions,
            rewards=self.rewards,
            next_seqs=self.next_seqs,
            dones=self.dones
            )
            self.file_idx = (self.file_idx + 1) % self.max_files_num
            self.crt_idx = 0
            self.reset()

    def sample(self, batch_size):
        num_files = len(os.listdir('./replay_data/'))
        rd_idx = np.random.choice(num_files*self.replay_memory_size, batch_size)
        file_names = list(map(lambda x: str(x//self.replay_memory_size), rd_idx))
        idx_in_file = list(map(lambda i: i % self.replay_memory_size, rd_idx))
        
        batch_seqs = np.zeros((batch_size, self.frame_size, self.frame_size, self.agent_history_length), dtype=np.float32)
        batch_actions = np.zeros(batch_size, np.uint8)
        batch_rewards = np.zeros(batch_size, np.float32)
        batch_next_seqs = np.zeros((batch_size, self.frame_size, self.frame_size, self.agent_history_length), dtype=np.float32)
        batch_dones =np.zeros(batch_size, np.bool)

        for i, file, idx in enumerate(zip(file_names, idx_in_file)):
            tmp = np.load('./replay_data/'+file+'.npz')
            batch_seqs[i] = tmp['seqs'][idx]
            batch_actions[i] = tmp['actions'][idx]
            batch_rewards[i] = tmp['rewards'][idx]
            batch_next_seqs[i] = tmp['next_seqs'][idx]
            batch_dones[i] = tmp['dones'][idx]
            tmp.close()

        return batch_seqs, batch_actions, batch_rewards, batch_next_seqs, batch_dones

    def reset(self):
        self.seqs = np.zeros((self.replay_memory_size, self.frame_size, self.frame_size, self.agent_history_length), dtype=np.float32)
        self.actions = np.zeros(self.replay_memory_size, np.uint8)
        self.rewards = np.zeros(self.replay_memory_size, np.float32)
        self.next_seqs = np.zeros((self.replay_memory_size, self.frame_size, self.frame_size, self.agent_history_length), dtype=np.float32)
        self.dones = np.zeros(self.replay_memory_size, np.bool)