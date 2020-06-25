import gym
from agent import Agent
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], True
)

def main():

    # environment
    env_name = 'Boxing-v4'
    env = gym.make(env_name)

    agent = Agent(env)

    agent.test('./save_weights/dqn_boxing_400epi.h5')

if __name__ == "__main__":
    main()
