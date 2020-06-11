import gym
from dqn_agent import DQNAgent
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], True
)

def main():

    # environment
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    agent = DQNAgent(env)

    agent.test()

if __name__ == "__main__":
    main()