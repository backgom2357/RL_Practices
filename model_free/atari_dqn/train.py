import gym
from agent import Agent
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], True
)

def main():

    # hyperparameter
    max_episode_num = 3000

    # environment
    env_name = 'Boxing-v0'
    env = gym.make(env_name)

    agent = Agent(env)

    # train
    agent.train(max_episode_num)

    # result
    agent.plot_result()

if __name__ == "__main__":
    main()
