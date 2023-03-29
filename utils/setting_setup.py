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

    # --------------------- model config
    parser.add_argument("--encoder", type=str, default="dense",
                        help="model type CVAE")

    parser.add_argument("--shared-size", type=list, default=[784, 784],
                        help="shared network layer size")
    parser.add_argument("--en_size", type=list, default=[784, 784, 512, 256],
                        help="encoder layer size")
    parser.add_argument("--de_size", type=list, default=[256, 512, 784, 784],
                        help="decoder layer size")

    parser.add_argument("--s-latent-size", type=int, default=784,
                        help="shared network latent size")
    parser.add_argument("--latent-size", type=int, default=128,
                        help="Embedding vector size")

    # --------------------- train config
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-6,
                        help="weight decay")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Data batch size")
    parser.add_argument("--scaler", type=bool, default=False,
                        help="Trigger the torch scaler function")
    # --------------------- meta config
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--task_num', type=int, default=10,
                        help='number of tasks (should be equal to number of classes/clusters)')
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for fine-tunning', default=10)
    # --------------------- pretrain config
    parser.add_argument("--pretrain-iterations", type=int,
                        help="number of pretraining iterations", default=0)
    parser.add_argument("--meta-train-iterations", type=int,
                        help="number of meta-training iterations", default=1000)
    # --------------------- loss config

    return parser.parse_args()
