from dqn.agent import Agent
import config
import gym


env = gym.make("Boxing-v4")
config = config.Config()

agent = Agent(config, env)

agent.play()