class Config():
    def __init__(self):
        """
        hyperparameters
        """
        self.learning_rate = 0.001
        self.gradient_momentum = 0.95
        self.initial_exploration = 1.0
        self.final_exploration = 0.1
        self.replay_memory_size = 50000
        self.replay_start_size = 5000
        self.final_exploration_frame = 250
        self.epsilon = self.initial_exploration

        self.frame_size = 84 # 크기를 키우면 allocate memory problem이 난다.
        self.batch_size = 34
        self.discount_factor = 0.99
        self.target_network_update_frequency = 7
        self.agent_history_length = 4
        self.update_frequency = 4
        self.skip_frames = 4
        self.tau = 0.6