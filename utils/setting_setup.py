import argparse
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser()
    # run--config
    parser.add_argument("--seed", type=int, default=730,
                        help="one manual random seed")
    parser.add_argument("--n-seed", type=int, default=1,
                        help="number of runs")

    # --------------------- Path
    parser.add_argument("--data-dir", type=Path, default="D:/Datasets/",
                        help="Path to the mnist dataset")
    parser.add_argument("--exp-dir", type=Path, default="D:/Github/1-RepresentationLearning/IVAE/experiments",
                        help="Path to the experiment folder, where all logs/checkpoints will be stored")

    # --------------------- flag & name
    parser.add_argument("--mode", type=str, default="train",
                        help="experiment mode")
    parser.add_argument("--log-delay", type=float, default=2.0,
                        help="Time between two consecutive logs (in seconds)")
    parser.add_argument("--eval", type=bool, default=True,
                        help="Evaluation Trigger")
    parser.add_argument("--log-flag", type=bool, default=False,
                        help="Logging Trigger")
    parser.add_argument("--f-cluster", type=bool, default=True,
                        help="Trigger the clustering to get salient feature of specific categories")

    # --------------------- Environment config
    parser.add_argument("--noise", type=float, default=0.01,
                        help="network noise")
    parser.add_argument("--lamda", type=float, default=1,
                        help="channel gain coefficient")
    parser.add_argument("--user-num", type=int, default=10,
                        help="number of users")
    parser.add_argument("--lamda", type=str, default=0.001,
                        help="signal wave length")

    return parser.parse_args()
