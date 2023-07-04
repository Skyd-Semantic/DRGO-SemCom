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

class ResultManager:
    def __init__(self):
        self.dictionary = {}

    def add_setting(self, setting_name, values):
        setting_dict = {}
        for value in values:
            key_dict = {
                'Transmission Time': None,
                'Power': None,
                'Transforming Factor': None,
                'Number of Channels': None
            }
            setting_dict[value] = key_dict
        self.dictionary[setting_name] = setting_dict

    def remove_setting(self, setting_name):
        if setting_name in self.dictionary:
            del self.dictionary[setting_name]
        else:
            print(f"Setting '{setting_name}' does not exist in the dictionary.")

    def update_setting_value(self, setting_name, value, transmission_time=None, power=None, transforming_factor=None, num_channels=None):
        if setting_name in self.dictionary:
            setting_dict = self.dictionary[setting_name]
            if value in setting_dict:
                key_dict = setting_dict[value]
                if transmission_time is not None:
                    key_dict['Transmission Time'] = transmission_time
                if power is not None:
                    key_dict['Power'] = power
                if transforming_factor is not None:
                    key_dict['Transforming Factor'] = transforming_factor
                if num_channels is not None:
                    key_dict['Number of Channels'] = num_channels
            else:
                print(f"Value '{value}' does not exist for setting '{setting_name}'.")
        else:
            print(f"Setting '{setting_name}' does not exist in the dictionary.")

    def get_setting_value(self, setting_name, value):
        if setting_name in self.dictionary:
            setting_dict = self.dictionary[setting_name]
            if value in setting_dict:
                return setting_dict[value]
            else:
                print(f"Value '{value}' does not exist for setting '{setting_name}'.")
        else:
            print(f"Setting '{setting_name}' does not exist in the dictionary.")
        return None

    def get_all_settings(self):
        return self.dictionary

def save_item(self, item_actor, item_critic, item_name):
    if not os.path.exists(self.save_folder_name):
        os.makedirs(self.save_folder_name)
    torch.save(item_actor, os.path.join(self.save_folder_name, "actor-" + item_name + ".pt"))
    torch.save(item_critic, os.path.join(self.save_folder_name, "critic-" + item_name + ".pt"))


def load_item(self, item_name):
    return torch.load(os.path.join(self.save_folder_name, "actor-" + item_name + ".pt")), \
           torch.load(os.path.join(self.save_folder_name, "critic-" + item_name + ".pt"))
