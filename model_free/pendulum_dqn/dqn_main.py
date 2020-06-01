import gym
from dqn_agent import DQNAgent

def main():
    max_episode_num = 3000
    env_name = 'MountainCar-v0'

    env = gym.make(env_name)
    agent = DQNAgent(env)

    # train
    agent.train(max_episode_num)

    # result
    agent.plot_result()

if __name__ == "__main__":
    main()