import gym
from agent import Agent
import tensorflow as tf
from neural_net import *
import cv2
import numpy as np
from utils import *
import matplotlib
import matplotlib.pyplot as plt
from config import Config

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], True
)

cf = Config()

def test(env_name, weights_url, render=False, check_input_frames=False, check_log_plot=False):

    # matplotlib 설정
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()

    # environment
    env_name = env_name
    env = gym.make(env_name)
    
    action_dim = env.action_space.n
    
    model = build_duel_dqn_model(cf.frame_size, action_dim, cf.agent_history_length)
    model.load_weights(weights_url)
    # print(model.trainable_weights)

    # initialize sequence
    # reset env and observe initial state
    done = 0
    initial_frame = env.reset()
    seq = [preprocess(initial_frame)]
    for _ in range(cf.agent_history_length - 1):
        obs, _, _, _ = env.step(0)
        seq.append(preprocess(obs,crop=(0,210,0,160)))
    seq = np.stack(seq, axis=3)
    seq = np.reshape(seq, (1, cf.frame_size, cf.frame_size, cf.agent_history_length))

    keep_action = 0
    
    frame = 0

    while not done:

        frame += 1

        # # render
        if render:
            env.render()

        # get action
        action = np.argmax(model(normalize(seq))[0])

        # # frame skip hard coding
        # if frame % 3 != 0:
        #     _, _, _, _ = env.step(keep_action)
        #     continue
        # keep_action = action

        # observe next frame
        observation, reward, done, info = env.step(action)
        # preprocess for next sequence
        next_seq = np.append(preprocess(observation, crop=(0,210,0,160)), seq[..., :3], axis=3)
        # store transition in replay memory
        seq = next_seq

        # # check what the agent see
        if check_input_frames:
            test_img = np.reshape(next_seq, (84, 84, 4))
            test_img = cv2.resize(test_img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
            cv2.imshow('obs', test_img)
            cv2.waitKey(0)!=ord('l')
            if cv2.waitKey(25)==ord('q') or done:
                cv2.destroyAllWindows()
        
        # check log & plot
        if check_log_plot:
            q = np.array(model(normalize(next_seq))[0])
            plot_durations(q, is_ipython)
            # print(action, q, end='\r')
            print(action, max(q), min(q), sum(q)/action_dim, end='\r')

    plt.ioff()


def plot_durations(q, is_ipython):
    ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
    }

    plt.figure(2)
    plt.clf()
    plt.xlabel('Actions')
    plt.ylabel('Q value')
    action = np.argmax(q)
    mean = np.mean(q)
    normalized_q = (q - mean)/2
    plt.title(ACTION_MEANING[action])
    plt.ylim(-1, 1)
    x = np.arange(len(q))
    xlabel = [str(a) for a in range(len(q))]
    color = ['hotpink' if i==action else 'c' for i in range(len(q))]
    plt.bar(x, normalized_q, color=color)
    plt.xticks(x, xlabel)
    plt.pause(0.01)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

if __name__ == "__main__":
    env_setting = [cf.ATARI_GAMES[1], './save_weights/dqn_boxing_3600epi.h5']
    test(*env_setting, render=True, check_log_plot=True)
