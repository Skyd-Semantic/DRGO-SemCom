from agents.ddpg.agent import *
from utils.setting_setup import *
from envs.environment import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_arguments()

    env = DRGO_env(args)

    agent = DDPGAgent(
        args,
        env
    )
    agent.train(args)
    # agent.evaluate(args)
