import argparse
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser()
    """ ======================================================== """
    """ ====================== Run config ===================== """
    """ ======================================================== """
    parser.add_argument("--seed", type=int, default=730,
                        help="one manual random seed")
    parser.add_argument("--n-seed", type=int, default=1,
                        help="number of runs")

    # --------------------- Path
    parser.add_argument("--data-dir", type=Path, default="D:/Datasets/",
                        help="Path to the mnist dataset")
    parser.add_argument("--model-dir", type=Path, default="./results/models/",
                        help="Path to the experiment folder, where all logs/checkpoints will be stored")
    parser.add_argument("--result-path", type=Path, default="./results/exp_evals/results.pkl",
                        help="Path to the experimental evaluation results")

    """ ======================================================== """
    """ ====================== Flag & name ===================== """
    """ ======================================================== """
    parser.add_argument("--mode", type=str, default="train",
                        help="experiment mode")
    parser.add_argument("--log-delay", type=float, default=2.0,
                        help="Time between two consecutive logs (in seconds)")
    parser.add_argument("--eval", type=bool, default=True,
                        help="Evaluation Trigger")
    parser.add_argument("--log-flag", type=bool, default=False,
                        help="Logging Trigger")
    parser.add_argument("--plot-interval", type=int, default=5000,
                        help="Number of step needed to plot new accuracy plot")
    parser.add_argument("--save-flag", type=bool, default=True,
                        help="Save Trigger")

    """ ======================================================== """
    """ ================== Environment config ================== """
    """ ======================================================== """
    parser.add_argument("--noise", type=float, default=0.01,
                        help="network noise")
    parser.add_argument("--user-num", type=int, default=10,
                        help="number of users")
    parser.add_argument("--lamda", type=float, default=0.001,
                        help="signal wave length")
    parser.add_argument("--poweru-max", type=float, default=10,
                        help="max power of user threshold")
    parser.add_argument("--bandwidth", type=float, default=100,
                        help="signal bandwidth")

    """ ======================================================== """
    """ ================== Semantic AI config ================== """
    """ ======================================================== """
    parser.add_argument("--L", type=float, default=50,
                        help="Lipschitz smooth variables")

    """ ======================================================== """
    """ ===================== Agent config ===================== """
    """ ======================================================== """
    parser.add_argument("--drl-algo", choices=['ddpg-ei', 'ddpg'],
                        default='ddpg-ei'
                        help="choice of DRL algorithm")
    parser.add_argument("--memory-size", type=int, default=100000,
                        help="size of the replay memory")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="data batch size")
    parser.add_argument("--ou-theta", type=float, default=1.0,
                        help="ou noise theta")
    parser.add_argument("--ou-sigma", type=float, default=0.1,
                        help="ou noise sigma")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--initial-steps", type=int, default=1e4,
                        help="initial random steps")
    parser.add_argument("--tau", type=float, default=5e-3,
                        help="initial random steps")
    parser.add_argument("--max-episode", type=int, default=100,
                        help="max episode")
    parser.add_argument("--max-step", type=int, default=500,
                        help="max number of step per episode")
    parser.add_argument("--max-episode-eval", type=int, default=5,
                        help="max evaluation episode")
    parser.add_argument("--max-step-eval", type=int, default=200,
                        help="max number of evaluation step per episode")
    parser.add_argument("--semantic-mode", type=str, default="learn",
                        help="learn | infer")
    parser.add_argument("--pen-coeff", type=float, default=0,
                        help="coefficient for penalty")
    parser.add_argument("--lr-actor", type=float, default=3e-4,
                        help="learning rate for actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3,
                        help="learning rate for critic")

    return parser.parse_args()
