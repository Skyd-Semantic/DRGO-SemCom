import os
import h5py
import torch
import pickle
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

class ResultManager:
    """
    Settings:
    -   Compression Ratio
    -   Noise Level
    -   Distortion Coefficient
    -   Number of users
    -   Transmission Time
    -   Power
    -   Transforming factor
    -   Number of channels
    """
    def __init__(self,
                 data_path):
        init_data = {
            'Compression Ratio': [],
            'Noise Level': [],
            'Distortion': [],
            'Number of Users': [],
            'Transmission Time': [],
            'Power': [],
            'Transforming Factor': [],
            'Number of Channels': []
        }
        self.result_df = pd.DataFrame(init_data)
        self.data_path = data_path
        if os.path.exists(self.data_path):
            pass
        else:
            self.df2pickle()

    def update_setting_value(self,
                             compression=None,
                             noise_lvl=None,
                             distortion_coeff=None,
                             user_num=None,
                             transmission_time=None,
                             power=None,
                             transforming_factor=None,
                             num_channels=None):
        new_data = {
            'Compression Ratio': [compression],
            'Noise Level': [noise_lvl],
            'Distortion': [distortion_coeff],
            'Number of Users': [user_num],
            'Transmission Time': [transmission_time],
            'Power': [power],
            'Transforming Factor': [transforming_factor],
            'Number of Channels': [num_channels]
        }
        new_df = pd.DataFrame(new_data)
        load_df = self.pickle2df()
        self.result_df = pd.concat([load_df, new_df], ignore_index=True)
        self.df2pickle()

    def query2draw(self, key_name, key_list, key_draw_1, key_draw_2):
        for key_val in key_list:
            filtered_df = self.result_df(self.result_df[key_name] == key_val)
            # draw it

    def df2pickle(self):
        with open(self.data_path, 'wb') as file:
            pickle.dump(self.result_df, file)

    def pickle2df(self):
        with open(self.data_path, 'rb') as file:
            target_df = pickle.load(file)
        return target_df

    def get_value(self):
        return self.result_df
def save_item(self, item_actor, item_critic, item_name):
    if not os.path.exists(self.save_folder_name):
        os.makedirs(self.save_folder_name)
    torch.save(item_actor, os.path.join(self.save_folder_name, "actor-" + item_name + ".pt"))
    torch.save(item_critic, os.path.join(self.save_folder_name, "critic-" + item_name + ".pt"))

def load_item(self, item_name):
    return torch.load(os.path.join(self.save_folder_name, "actor-" + item_name + ".pt")), \
           torch.load(os.path.join(self.save_folder_name, "critic-" + item_name + ".pt"))
