import gym
from agent import Agent
import tensorflow as tf

def main():

    # hyperparameter
    max_episode_num = 8000


    # environment
    env_name = 'Pong-v0'
    env = gym.make(env_name)

    # state dimension
    state_dim = env.observation_space.shape[0]
    # action dimension
    action_dim = env.action_space.n

    agent = Agent(env, state_dim, action_dim)

    # train
    agent.train(max_episode_num)

if __name__ == "__main__":
    main()
