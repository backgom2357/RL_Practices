import gym
from dqn_agent import DQNAgent
import tensorflow as tf

def main():

    # environment
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    agent = DQNAgent(env)

    agent.test('./save_weights/mountainCar700epi_dqn.h5')

if __name__ == "__main__":
    main()