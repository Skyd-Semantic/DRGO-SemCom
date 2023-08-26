from agents.ddpg.agent import *
from utils.setting_setup import *
from envs.environment import *
from beautifultable import BeautifulTable

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_arguments()

    table = BeautifulTable(maxwidth=140, detect_numerics=False)
    table.rows.append(["AI Network", args.ai_network, "Algorithm", args.drl_algo, "Plot Interval", args.plot_interval])
    table.rows.append(["Sem Mode", args.semantic_mode, "Noise", args.noise, "Learning Rate", args.lr_critic])
    table.rows.append(["Power", args.poweru_max, "AI LR", args.ai_lr, "Lipschitz", args.L])
    table.rows.append(["Pen Coeff", args.pen_coeff, "Bandwidth", args.bandwidth, "Num. User", args.user_num])

    print(table)

    env = DRGO_env(args)

    agent = DDPGAgent(
        args,
        env
    )
    agent.train(args)
    # agent.evaluate(args)
