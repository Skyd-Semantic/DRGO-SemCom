import os
import h5py
import torch


def save_results(args,
                 scores: List[float],
                 actor_losses: List[float],
                 critic_losses: List[float],
                 reward_list: List[float]
                 ):
    algo = args.algorithm + "-" + args.max_episode + "-" + args.max_step + "-" + args.user_num + "-" args.pen_coeff
    result_path = "./results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if len(scores):
        algo = algo
        file_path = result_path + "{}.h5".format(algo)
        print("File path: " + file_path)

        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('reward', data=reward_list)
            hf.create_dataset('scores', data=scores)
            hf.create_dataset('actor_losses', data=actor_losses)
            hf.create_dataset('critic_losses', data=critic_losses)

def save_item(self, item, item_name):
    if not os.path.exists(self.save_folder_name):
        os.makedirs(self.save_folder_name)
    torch.save(item, os.path.join(self.save_folder_name, item_name + ".pt"))

def load_item(self, item_name):
    return torch.load(os.path.join(self.save_folder_name, item_name + ".pt"))