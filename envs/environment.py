import math

import numpy as np
from utils.setting_setup import *
import scipy

from envs.env_utils import *
from envs.env_agent_utils import *


class DRGO_env(env_utils, env_agent_utils):
    def __init__(self, args):
        # Network setting
        self.penalty = 0
        self.noise = args.noise
        self.lamda = args.lamda
        self.N_User = args.user_num
        self.G_CU_list = np.ones((self.N_User, 1))  # User directivity
        self.G_BS_t = 1  # BS directivity
        self.Z_u = 10000  # Data size
        self.Num_BS = 1  # Number of Base Stations
        self.max_step = args.max_step
        self.drl_algo = args.drl_algo

        # Power setting
        self.P_u_max = args.poweru_max
        self.eta = args.ai_lr
        self.naught = 3.9811 * (10 ** (-21))  # -174 dBm/Hz -> W/Hz
        # Bandwidth
        self.B = args.bandwidth

        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = 1
        self.BS_R_min = 0.1

        # Goal-oriented Settings
        # Learn
        self.acc_threshold = 0.05
        self.Lipschitz = args.L
        # Inference
        self.inf_capacity = 0.9
        self.pen_coeff = args.pen_coeff

        self.sigma_data = 0.01
        self.semantic_mode = args.semantic_mode
        self.OSigmaMapping = {
            'Comp. Ratio': [192, 96, 48, 32, 24, 16, 12, 10, 9, 8, 6, 4, 3, 2],
            'Loss': [0.799, 0.539, 0.337, 0.249, 0.193, 0.131, 0.098, 0.079, 0.069, 0.060, 0.034, 0.020, 0.017, 0.015]
        }

        """ =============== """
        """     Actions     """
        """ =============== """
        self.o = np.random.randint(0, self.N_User, size=[self.N_User, 1])
        # tau is Sub-carrier-Allocation. It is an array with form of Num_Nodes interger number,
        # value change from [0:Num_sub-1] (0 means Sub#1)
        self.tau = np.random.randint(0, self.N_User, size=[self.N_User, 1])
        # eta is AP-Allocation. It is an array with form of Num_Nodes interger number,
        # value change from [0:Num_APs-1] (0 means Sub#1)
        self.P_n = np.reshape((np.random.rand(1, self.N_User) * self.P_u_max), (self.N_User, 1))

        """ ========================================= """
        """ ===== Function-based Initialization ===== """
        """ ========================================= """
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)

        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        self.commonDataRate = self._calculateDataRate(self.ChannelGain)
        self.T = 0

        """ ============================ """
        """     Environment Settings     """
        """ ============================ """
        self.rewardMatrix = np.array([])
        self.observation_space = self._wrapState().squeeze()
        self.action_space = self._wrapAction()
        print(f"O Space: {np.shape(self.observation_space)} | A Space: {np.shape(self.action_space)}")

    def step(self, action, step):
        self.tau, self.o, self.P_n = self._decomposeAction(action)
        # Environment change
        self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self.User_trajectory + self.U_location
        # State wrap
        state_next = self._wrapState()
        # Re-calculate channel gain
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)

        self.T = self._Time()  # Generate self.T
        # Calculate distortion rate
        sigma_data = self.sigma_data

        # We assume that it follows CLT
        # This function can be changed in the future
        temp_c = 20
        # sigma_sem = np.exp(temp_c * (1 - self.o) ** 2)
        # print(f"sem: {np.shape(sigma_sem)}")
        sigma_sem, self.o_fixed = self._OSigmaMapping()
        # print(f"sem2: {np.shape(sigma_sem)}|{sigma_sem}|{self.o}")
        # sigma_tot_sqr = 1 / ((1 / sigma_sem ** 2) + (1 / sigma_data ** 2))
        sigma_tot_sqr = (sigma_sem ** 2) + (sigma_data ** 2)
        self.sigma_tot_sqr = sigma_tot_sqr
        self.sigma_sem = sigma_sem
        # Goal-oriented penalty
        if self.semantic_mode == "learn":
            penalty = max(np.sum((self.eta ** 2 * self.Lipschitz / 2 - self.eta) * \
                                 (self.Lipschitz ** 2) * sigma_tot_sqr - self.acc_threshold), 0)
            self.penalty = penalty
        else:
            penalty = max(np.sum(
                (1 / math.sqrt(2 * math.pi)) * self.inf_capacity * np.exp(-1 / (4 * (self.B ** 2) * sigma_tot_sqr))), 0)
        if self.drl_algo == "ddpg-ei":
            pass
        else:
            penalty += 2 * np.max(np.sum(self.tau[0]) - 1, 0)  # if tau>1 -> add tau-1 to penalty
            penalty -= 2 * np.min(np.sum(self.tau[0]) - 0, 0)  # if tau<0 -> add -tau to penalty
            penalty += 2 * sum([max(i - 1, 0) for i in self.o[0]])  # o > 1 -> add (o-1) to penalty
            penalty -= 2 * sum([min(i - 0, 0) for i in self.o[0]])  # o < 0 -> add -o to penalty
            penalty += 2 * sum([max(i - 1, 0) for i in self.P_n[0]])  # Pn < 0 -> add -Pn to penalty
            penalty -= 2 * sum([max(i - 1, 0) for i in self.P_n[0]])  # Pn > 0 -> add (Pn-1) to penalty

        reward =  - self.T - self.pen_coeff * penalty
        # print(f"step: {step} --> rew: {reward} | T: {self.T}| pena: {penalty}")
        """
        T = 100 
        - Normally, the constraint * penalty should be around 0.01 - 0.2 of T
        - Print and observe the distribution of the constraints -> decide the alpha
        """
        if step == self.max_step:
            done = True
        else:
            done = False

        info = None
        return state_next, reward, done, info

    def step_eval(self, action, step):
        """

        :return:
        """
        self.tau, self.o, self.P_n = self._decomposeAction(action)
        # Environment change
        self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self.User_trajectory + self.U_location
        # State wrap
        state_next = self._wrapState()
        # Re-calculate channel gain
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)

        self.T = self._Time()  # Generate self.T
        # Calculate distortion rate
        sigma_data = self.sigma_data

        # We assume that it follows CLT
        # This function can be changed in the future
        temp_c = 20
        sigma_sem = np.exp(temp_c * (1 - self.o) ** 2)
        sigma_tot_sqr = 1 / ((1 / sigma_sem ** 2) + (1 / sigma_data ** 2))

        # Goal-oriented penalty
        if self.semantic_mode == "learn":
            penalty = max(np.sum((self.eta ** 2 * self.Lipschitz / 2 - self.eta) * \
                                 (self.Lipschitz ** 2) * sigma_tot_sqr - self.acc_threshold), 0)
        else:
            penalty = max(np.sum(
                (1 / math.sqrt(2 * math.pi)) * self.inf_capacity * np.exp(-1 / (4 * (self.B ** 2) * sigma_tot_sqr))), 0)

        reward = - self.T - self.pen_coeff * penalty
        print(f"step: {step} --> rew: {reward} | T: {self.T}| pena: {penalty}")

        if step == self.max_step:
            done = True
        else:
            done = False
        info = None

        return state_next, [reward, self.T, sigma_tot_sqr, ], done, info

    def reset(self):
        # Base station initialization
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)

        # Use initialization
        # self.U_location = np.expand_dims(self._location_CU_Generator(), axis=0)
        # self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        # Distance calculation
        self.distance_CU_BS = self._distance_Calculated(self.BS_location, self.U_location)

        # re-calculate channel gain
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)

        # Generate next state [set of ChannelGain]
        state_next = self._wrapState()
        return state_next


if __name__ == '__main__':
    args = get_arguments()
    env = DRGO_env(args)
