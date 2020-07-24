import gym
from agent import Agent
import tensorflow as tf
from neural_net import DQN
import cv2
import numpy as np
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], True
)

agent_history_length = 4
frame_size = 84

def preprocess(frame):
    frame = np.reshape(cv2.resize(frame[0:188, 23:136, :], dsize=(frame_size, frame_size))[..., 0],
                        (1, frame_size, frame_size, 1))
    return np.array(frame, dtype=np.float32) / 255

def main():

    # environment
    env_name = 'Pong-v0'
    env = gym.make(env_name)
    
    action_dim = 6
    
    dqn = DQN(frame_size, action_dim, agent_history_length)
    model = dqn.model
    model.load_weights('./save_weights/dqn_pong_1500epi.h5')

    # initialize sequence
    # reset env and observe initial state
    done = 0
    initial_frame = env.reset()
    seq = [preprocess(initial_frame)]
    for _ in range(agent_history_length - 1):
        obs, _, _, _ = env.step(0)
        seq.append(preprocess(obs))
    seq = np.stack(seq, axis=3)
    seq = np.reshape(seq, (1, frame_size, frame_size, agent_history_length))

    while not done:

        time.sleep(0.02)

        # # render
        env.render()
        # get action
        action = np.argmax(model(seq)[0])
        # observe next frame
        observation, reward, done, info = env.step(action)
        # preprocess for next sequence
        next_seq = np.append(preprocess(observation), seq[..., :3], axis=3)
        # store transition in replay memory
        seq = next_seq

        # # check what the agent see
        # test_img = np.reshape(next_seq, (84, 84, 4))
        # test_img = cv2.resize(test_img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        # cv2.imshow('obs', test_img)
        # if cv2.waitKey(25)==ord('q') or done:
        #     cv2.destroyAllWindows()
        q = model(seq)[0]
        print(action, np.max(q), np.min(q), end='\r')


if __name__ == "__main__":
    main()
