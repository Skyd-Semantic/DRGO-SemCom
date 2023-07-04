import os
import h5py
import torch
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def save_item(self, item_actor, item_critic, item_name):
    if not os.path.exists(self.save_folder_name):
        os.makedirs(self.save_folder_name)
    torch.save(item_actor, os.path.join(self.save_folder_name, "actor-" + item_name + ".pt"))
    torch.save(item_critic, os.path.join(self.save_folder_name, "critic-" + item_name + ".pt"))

def load_item(self, item_name):
    return torch.load(os.path.join(self.save_folder_name, "actor-" + item_name + ".pt")), \
           torch.load(os.path.join(self.save_folder_name, "critic-" + item_name + ".pt"))

class ResultManager:
    """
    Settings:
    -   Noise Level
    -   Distortion Coefficient
    -   Number of users
    Key:
    -   Transmission Time
    -   Power
    -   Transforming factor
    -   Number of channels
    """

    def __init__(self,
                 data_path):
        init_data = {
            'Noise Level': [],
            'Distortion': [],
            'Number of Users': [],
            'Transmission Time': [],
            'Power': [],
            'Transforming Factor': [],
            'Number of Channels': []
        }
        self.colorset = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.result_df = pd.DataFrame(init_data)
        self.data_path = data_path
        if os.path.exists(self.data_path):
            pass
        else:
            self.df2pickle()

    def update_setting_value(self,
                             noise_lvl=None,
                             distortion_coeff=None,
                             user_num=None,
                             transmission_time=None,
                             power=None,
                             transforming_factor=None,
                             num_channels=None):
        new_data = {
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

    def query2draw(self, key_name, key_list, key_draw_x, key_draw_y):
        result_df = self.pickle2df()
        sns.set_style("whitegrid")  # Turn on the grid
        sns.lineplot(
            x=key_draw_x,
            y=key_draw_y,
            data=result_df,
            hue=key_name,
            linewidth=1.5,
            linestyle='dashed',
            palette=self.colorset
        )
        sns.scatterplot(x=key_draw_x,
                        y=key_draw_y,
                        hue=key_name,
                        data=result_df,
                        style=key_name,
                        palette=self.colorset,
                        legend=False)

        values = result_df[key_name].unique()
        legend_labels = [f'o={value}' for value in values]
        print(legend_labels)
        plt.legend(labels=legend_labels,
                   loc='upper left',
                   ncol=2,
                   title=key_name)
        plt.savefig('plot.pdf', dpi=300)  # Save as PDF

    def df2pickle(self):
        with open(self.data_path, 'wb') as file:
            pickle.dump(self.result_df, file)

    def pickle2df(self):
        with open(self.data_path, 'rb') as file:
            target_df = pickle.load(file)
        return target_df

    def get_value(self):
        return self.result_df