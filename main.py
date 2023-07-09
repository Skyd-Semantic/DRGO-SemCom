from agents.ddpg.agent import *
from utils.setting_setup import *
from envs.environment import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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
