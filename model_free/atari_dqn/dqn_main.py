import gym
from dqn_agent import Agent

def main():

    # hyperparameter
    max_episode_num = 2000

    # environment
    env_name = 'Boxing-v4'
    env = gym.make(env_name)

    agent = Agent(env)

    # train
    agent.train(max_episode_num)

    # # result
    # agent.plot_result()

if __name__ == "__main__":
    main()
