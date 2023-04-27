import math

import numpy as np
from utils.setting_setup import *
import scipy

from envs.env_utils import *
from envs.env_agent_utils import *

class DRGO_env(env_utils, env_agent_utils):
    def __init__(self, args):
        # Network setting
        self.noise = args.noise
        self.lamda = args.lamda
        self.N_User = args.user_num
        self.G_CU_list = np.ones((self.N_User, 1))  # User directivity
        self.G_BS_t = 1  # BS directivity
        self.Z_u = 10000  # Data size
        self.Num_BS = 1  # Number of Base Stations
        self.N_User = 10  # Number of Users

        # Power setting
        self.P = args.power
        self.P_u_max = args.poweru_max
        self.P_0 = args.power0
        self.Pn = args.powern
        self.eta = 0.7  # de tinh R_u
        self.sigma = 3.9811*(np.e**(-21+7))                        # -174 dBm/Hz -> W/Hz
        # Bandwidth
        self.B = args.bandwidth

        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = 1
        self.BS_R_min = 0.1

        # Goal-oriented Settings
        self.acc_threshold = 0.05
        self.Lipschitz = 0.005
        self.inf_capacity = 0.9
        self.lamda = args.pen_coeff

        self.sigma_data = 0.01
        self.semantic_mode = args.semantic_mode

        """ =============== """
        """     Actions     """
        """ =============== """
        self.o = np.random.randint(0, self.N_User, size=[self.N_User,1])
        # tau is Sub-carrier-Allocation. It is an array with form of Num_Nodes interger number,
        # value change from [0:Num_sub-1] (0 means Sub#1)
        self.tau = np.random.randint(0, self.N_User, size=[self.N_User,1])
        # eta is AP-Allocation. It is an array with form of Num_Nodes interger number,
        # value change from [0:Num_APs-1] (0 means Sub#1)
        self.P_n = np.reshape((np.random.rand(1, self.N_User) * self.P_u_max), (self.N_User,1))

        """ ========================================= """
        """ ===== Function-based Initialization ===== """
        """ ========================================= """
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)

        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        self.commonDataRate = self._calculateDataRate(self.ChannelGain)
        self.T = 0                                           # initialize rewards)

        """ ============================ """
        """     Environment Settings     """
        """ ============================ """
        self.rewardMatrix = np.array([])
        self.observation_space = self._wrapState().squeeze()
        self.action_space = self._wrapAction()


    def step(self, action):
        self.tau, self.o, self.P_n = self._decomposeAction(action)
        # Environment change
        self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self.User_trajectory + self.U_location
        # State wrap
        state_next = self._wrapState()
        # Re-calculate channel gain
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)

        self.T = self._Time()    # Generate self.T
        # Calculate distortion rate
        sigma_data = self.sigma_data

        # We assume that it follows CLT
        # This function can be changed in the future
        temp_c = 20
        sigma_sem = np.exp(temp_c*(1-self.o)**2)
        sigma_tot_sqr = 1/((1/sigma_sem**2)+(1/sigma_data**2))

        # Goal-oriented penalty
        if self.semantic_mode == "learn":
            penalty = np.sum((self.eta**2 * self.Lipschitz/2 - self.eta)*\
                      (self.Lipschitz**2) * sigma_tot_sqr - self.acc_threshold)
        else:
            penalty = np.sum((1/math.sqrt(2*math.pi)) * self.inf_capacity * np.exp( -1/(4*(self.B**2)*sigma_tot_sqr) ))
        # print(f"penalty: {penalty}")
        reward = - self.T - self.lamda*penalty
        print(f"rew: {reward} | T: {self.T}| pena: {penalty}")
        """
        T = 100 
        - Normally, the constraint * penalty should be around 0.01 - 0.2 of T
        - Print and observe the distribution of the constraints -> decide the alpha
        """

        done = False
        info = None
        return state_next, reward, done, info

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

    def close(self):
        pass


if __name__ == '__main__':
    args = get_arguments()
    env = DRGO_env(args)