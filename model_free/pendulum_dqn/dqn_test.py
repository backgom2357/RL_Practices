import gym
from dqn_agent import Agent

def main():

    # environment
    env_name = 'Boxing-v0'
    env = gym.make(env_name)

    agent = Agent(env, is_test=True)

    # train
    agent.test()

    # # result
    # agent.plot_result()

if __name__ == "__main__":
    main()
