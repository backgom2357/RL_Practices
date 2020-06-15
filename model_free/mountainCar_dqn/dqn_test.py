import gym
from dqn_agent import DQNAgent
import tensorflow as tf

def main():

    # environment
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    agent = DQNAgent(env)

    agent.test()

if __name__ == "__main__":
    main()