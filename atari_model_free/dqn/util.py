import cv2
import numpy as np

class Util:
    def __init__(self, config):
        self.config = config

    def preprocess(self, frame):
        frame = np.reshape(cv2.resize(frame, dsize=(self.config.frame_height, self.config.frame_width))[..., 0], (1, self.config.frame_height, self.config.frame_width, 1))
        return np.array(frame, dtype=np.float16)
