import gym
from agent import Agent
import tensorflow as tf
from neural_net import build_model
import cv2
import numpy as np
import time
from utils import preprocess

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], True
)

agent_history_length = 4
frame_size = 84

def main():

    # environment
    env_name = 'Boxing-v0'
    env = gym.make(env_name)
    
    action_dim = env.action_space.n
    
    model = build_model(frame_size, action_dim, agent_history_length)
    model.load_weights('./save_weights/dqn_boxing_5500epi.h5')
    print(model.trainable_weights)

    # initialize sequence
    # reset env and observe initial state
    done = 0
    initial_frame = env.reset()
    seq = [preprocess(initial_frame)]
    for _ in range(agent_history_length - 1):
        obs, _, _, _ = env.step(0)
        seq.append(preprocess(obs,crop=(0,210,0,160)))
    seq = np.stack(seq, axis=3)
    seq = np.reshape(seq, (1, frame_size, frame_size, agent_history_length))
    
    frame = 0

    while not done:

        frame += 1

        time.sleep(0.02)

        # # render
        # env.render()
        # get action
        action = np.argmax(model(seq)[0])

        # need for breakout test
        if frame % 100==0:
            action = 1

        # observe next frame
        observation, reward, done, info = env.step(action)
        # preprocess for next sequence
        next_seq = np.append(preprocess(observation, crop=(0,210,0,160)), seq[..., :3], axis=3)
        # store transition in replay memory
        seq = next_seq

        # check what the agent see
        test_img = np.reshape(next_seq, (84, 84, 4))
        test_img = cv2.resize(test_img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        cv2.imshow('obs', test_img)
        if cv2.waitKey(25)==ord('q') or done:
            cv2.destroyAllWindows()
        q = np.array(model(next_seq)[0])
        kk = np.sum(next_seq)
        print(action, max(q), min(q), sum(q)/action_dim, end='\r')

    print(frame)


if __name__ == "__main__":
    main()
