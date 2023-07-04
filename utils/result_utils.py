import os
import h5py
import torch
from typing import List


def save_results(
                 scores: List[float],
                 actor_losses: List[float],
                 critic_losses: List[float],
                 reward_list: List[float],
                 algo
                 ):
    result_path = "./results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if len(scores):
        file_path = result_path + "{}.h5".format(algo)
        print("File path: " + file_path)

        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('reward', data=reward_list)
            hf.create_dataset('scores', data=scores)
            hf.create_dataset('actor_losses', data=actor_losses)
            hf.create_dataset('critic_losses', data=critic_losses)

"""
Feed STATE into environment + re-calculate again the attributes
- Total transmission time vs.  
    - Power
    - Transforming factor (bits/word)
    - Number of channels
"""
def result_dict():
    """
    Settings:
        -   Compression Ratio
        -   Channel Level
        -   Distortion Coefficient
        -   Number of users
    The dicts should including
    Key:
        -   Transmission Time
        -   Power
        -   Transforming factor
        -   Number of channels
    :return:
    """
    pass

class DictManager:
    def __init__(self):
        self.dictionary = {}

    def add_setting(self, compression_value, transmission_time, power, transforming_factor, num_channels):
        key_dict = {
            'Transmission Time': transmission_time,
            'Power': power,
            'Transforming Factor': transforming_factor,
            'Number of Channels': num_channels
        }
        self.dictionary['Compression'] = {compression_value: key_dict}

def save_item(self, item_actor, item_critic, item_name):
    if not os.path.exists(self.save_folder_name):
        os.makedirs(self.save_folder_name)
    torch.save(item_actor, os.path.join(self.save_folder_name, "actor-" + item_name + ".pt"))
    torch.save(item_critic, os.path.join(self.save_folder_name, "critic-" + item_name + ".pt"))


def load_item(self, item_name):
    return torch.load(os.path.join(self.save_folder_name, "actor-" + item_name + ".pt")), \
           torch.load(os.path.join(self.save_folder_name, "critic-" + item_name + ".pt"))
