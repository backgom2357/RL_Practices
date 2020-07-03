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

    def append(self, seq, action, reward, next_seq, done):
        self.seqs[self.crt_idx, ...] = seq
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_seqs[self.crt_idx, ...] = next_seq
        self.dones[self.crt_idx] = done

        self.crt_idx += 1

        if self.crt_idx == self.replay_memory_size:
            np.save('./replay_data/seqs/'+str(self.file_idx)+'.npy', self.seqs) 
            np.save('./replay_data/actions/'+str(self.file_idx)+'.npy', self.actions)
            np.save('./replay_data/rewards/'+str(self.file_idx)+'.npy', self.rewards)
            np.save('./replay_data/next_seqs/'+str(self.file_idx)+'.npy', self.next_seqs)
            np.save('./replay_data/dones/'+str(self.file_idx)+'.npy', self.dones)
            self.file_idx = (self.file_idx + 1) % self.max_files_num
            self.crt_idx = 0
            self.reset()

    def sample(self, batch_size):
        num_files = len(os.listdir('./replay_data/seqs'))
        rd_idx = np.random.choice(num_files*self.replay_memory_size, batch_size)
        file_names = list(map(lambda x: str(x//self.replay_memory_size), rd_idx))
        idx_in_file = list(map(lambda i: i % self.replay_memory_size, rd_idx))
        
        batch_seqs = np.zeros((batch_size, self.frame_size, self.frame_size, self.agent_history_length), dtype=np.float32)
        batch_actions = np.zeros(batch_size, np.uint8)
        batch_rewards = np.zeros(batch_size, np.float32)
        batch_next_seqs = np.zeros((batch_size, self.frame_size, self.frame_size, self.agent_history_length), dtype=np.float32)
        batch_dones =np.zeros(batch_size, np.bool)

        for i, (file, idx) in enumerate(zip(file_names, idx_in_file)):
            batch_seqs[i] = np.load('./replay_data/seqs'+file+'.npy')[idx]
            batch_actions[i] = np.load('./replay_data/actions'+file+'.npy')[idx]
            batch_rewards[i] = np.load('./replay_data/rewards'+file+'.npy')[idx]
            batch_next_seqs[i] = np.load('./replay_data/next_seqs'+file+'.npy')[idx]
            batch_dones[i] = np.load('./replay_data/dones'+file+'.npy')[idx]

        return batch_seqs, batch_actions, batch_rewards, batch_next_seqs, batch_dones

    def reset(self):
        self.seqs = np.zeros((self.replay_memory_size, self.frame_size, self.frame_size, self.agent_history_length), dtype=np.float32)
        self.actions = np.zeros(self.replay_memory_size, np.uint8)
        self.rewards = np.zeros(self.replay_memory_size, np.float32)
        self.next_seqs = np.zeros((self.replay_memory_size, self.frame_size, self.frame_size, self.agent_history_length), dtype=np.float32)
        self.dones = np.zeros(self.replay_memory_size, np.bool)