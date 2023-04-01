import numpy as np
from utils.setting_setup import *


class DRGO_env():
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
        self.P_BS_U = (self.P / self.N_User) * np.ones((self.N_User, 1))
        self.sigma = -10**(-18)                        # W/Hz
<<<<<<< HEAD
=======
        self.T = self._Time()
>>>>>>> f25795235e3fe8f389face313528ef4830898b6d
        # Bandwidth
        self.B = args.bandwidth

        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = 1
        self.BS_R_min = 0.1

        """ ========================================= """
        """ ===== Function-based Initialization ===== """
        """ ========================================= """
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        # self.User_trajectory = self._trajectory_U_Generator()
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)
        self.Pathloss = self._Pathloss_Calculated()
        self.H = self._channelGain_BS_CU()
        self.ChannelGain = self._ChannelGain_Calculated()
        self.commonDataRate = self._calculateDataRate(self.H)
        self.T = 0                                           # initialize rewards

        self.commonDataRate = self._calculateDataRate(self.H)

        # Channel Gain
        # self.H_CU = self._channelGain_BS_CU()

        """ =============== """
        """     Actions     """
        """ =============== """
        self.o = np.random.randint(0, self.N_User, size=[self.N_User,1])
        # tau is Sub-carrier-Allocation. It is an array with form of Num_Nodes interger number, value change from [0:Num_sub-1] (0 means Sub#1)
        self.tau = np.random.randint(0, self.N_User, size=[self.N_User,1])
        print(f"{np.shape(self.tau)}-{np.shape(self.tau)}")
        # self.beta = np.reshape(np.random.randint(0, self.N_User, size = self.N_User), self.N_User)
        # eta is AP-Allocation. It is an array with form of Num_Nodes interger number, value change from [0:Num_APs-1] (0 means Sub#1)

        self.P_n = np.random.randint(0, self.N_User, size=[self.N_User,1])

        """ ============================ """
        """     Environment Settings     """
        """ ============================ """
        self.rewardMatrix = np.array([])
        self.observation_space = self._wrapState().squeeze()
        self.action_space = self._wrapAction()

    def _channelGain_BS_CU(self):
        """     Free-space path los     """
        numerator = self.G_BS_t * self.G_CU_list * (self.lamda ** 2)  # Directivity_BS * Directivity_CU * lambda
        denominator = ((4 * np.pi) ** 3) * (self.distance_CU_BS ** 4)
        channelGain = numerator / denominator
        return channelGain

    def _location_BS_Generator(self):
        BS_location = [self.BS_x, self.BS_y]
        # print(BS_location)
        return np.array(BS_location)

    # Iot initialization
    def _location_CU_Generator(self):
        userList = []
        # hUser_temp = 1.65
        for i in range(self.N_User):
            r = self.BS_R_Range * np.sqrt(np.random.rand()) + self.BS_R_min
            theta = np.random.uniform(-np.pi, np.pi)
            xUser_temp = self.BS_x + r * np.cos(theta)
            yUser_temp = self.BS_y + r * np.sin(theta)
            userList.append([xUser_temp, yUser_temp])
            U_location = np.array(userList)
            # print(U_location)
        return U_location

    def _trajectory_U_Generator(self):
        userList = []
        for i in range(self.N_User):
            theta = 0
            theta = theta + np.pi / 360
            r = np.sin(theta)
            xUser_temp = r * np.cos(2 * theta)
            yUser_temp = r * np.sin(2 * theta)
            userList.append([xUser_temp, yUser_temp])
            User_trajectory = np.array(userList)
        return User_trajectory

    # def _trajectory_U_Generator(self):
    #   theta  = 0
    #   theta  = theta + np.pi/360
    #   r      = np.sin(theta)
    #   xUser_temp = r*np.cos(2*theta)
    #   yUser_temp = r*np.sin(2*theta)

    #   User_trajectory = [xUser_temp, yUser_temp]
    #   return np.array(User_trajectory)
    def _distance_Calculated(self, A, B):
        return np.array([np.sqrt(np.sum((A - B) ** 2, axis=1))]).transpose()

    # def _distance_Calculated(self):
    #       dist = np.zeros((self.Num_BS, self.N_User))
    #       for i in range(self.Num_BS):
    #           for j in range(self.N_User):
    #               dist[i][j] = np.sqrt(np.sum(self.BS_location[i]-self.U_location[j])**2)

    #       return dist

    def _Pathloss_Calculated(self):
        Pathloss = np.zeros((self.Num_BS, self.N_User))
        Pathloss = 10 ** (-2 * np.log10(4 * np.pi * self.distance_CU_BS / self.lamda))
        # print(Pathloss)
        return Pathloss

    def _ChannelGain_Calculated(self):
        numerator = self.G_BS_t * self.G_CU_list * (self.lamda ** 2)
        denominator = ((4 * np.pi) ** 3) * (self.distance_CU_BS ** 4)
        ChannelGain = numerator / denominator
        # print(ChannelGain)
        return np.array(ChannelGain)
<<<<<<< HEAD

    def _calculateDataRate(self, channelGain_BS_CU):
        sumCommonUserPower      = np.sum(self.P_BS_U)
        interferenceCommonUser  = ((channelGain_BS_CU))*sumCommonUserPower
        Numerator         = ((channelGain_BS_CU))*self.P_0

        interferenceBandwidth   = self.B * self.sigma
        Denominator       = interferenceCommonUser + interferenceBandwidth
        DataRate          = self.B * np.log2(1+(Numerator/Denominator))
        return DataRate

    def _Time(self):
        self.DataRate = self._calculateDataRate(self.H)

        print(f"{np.shape(self.o)} - {np.shape(self.tau)} - {np.shape(self.DataRate)}")
        T = (self.o * self.tau) / self.DataRate
        return np.sum(T)

=======
    def _calculateDataRate(self, channelGain_BS_CU):
        sumCommonUserPower      = np.sum(self.P_BS_U)
        interferenceCommonUser  = ((channelGain_BS_CU))*sumCommonUserPower
        commonNumerator         = ((channelGain_BS_CU))*self.P_0
        interferenceBandwidth   = self.B * self.sigma
        commonDenominator       = interferenceCommonUser + interferenceBandwidth
        commonDataRate          = self.B * np.log2(1+(commonNumerator/commonDenominator))
        print(f"interferenceBandwidth: {interferenceBandwidth}")
        print(f"interferenceCommonUser: {interferenceCommonUser}")
        print(f"commonNumerator: {commonNumerator}")
        print(f"commonDenominator: {commonDenominator}")
        print(f"commonDataRate:{commonDataRate}")
        return commonDataRate
    def _Time(self):
        self.commonDataRate = self._calculateDataRate(self.H)
        self.T = np.multiply(self.o, self.tau)/self.commonDataRate
        return T
>>>>>>> f25795235e3fe8f389face313528ef4830898b6d
    def _wrapState(self):
        self.H = self._channelGain_BS_CU()
        # print(np.shape(self.User_trajectory))
        state = np.concatenate((np.array(self.H).reshape(1, -1), np.array(self.U_location).reshape(1, -1),
                                np.array(self.User_trajectory).reshape(1, -1)), axis=1)
        # print(np.shape(state))
        # state = state.reshape(1,-1)
        # print(np.shape(state))
        return state

    def _decomposeState(self, state):
        H = state[0: self.N_User]
        U_location = state[self.N_User: 2 * self.N_User + 2]
        User_trajectory = state[self.N_User + 2: 2 * self.N_User + 4]
        return [
            np.array(H), np.array(U_location), np.array(User_trajectory)
            #   ]
            # def _decomposeState(self, state):
            #   H = state[:, :1]
            #   U_location = state[:, 1:3]
            #   User_trajectory = state[:, 3:]
            #   return [
            #       np.array(H), np.array(U_location), np.array(User_trajectory)
        ]

    def _wrapAction(self):
        action = np.concatenate((np.array(self.tau),
                                 np.array(self.o),
                                 np.array(self.P_n)), axis=1)

        # print(f'wrap-action: {action}')
        return action

    def _decomposeAction(self, action):
        print(action)
        tau = action[:,0]
        o = action[:,1]
        P_n = action[:,2]

        print(f"======================")
        print(f"tau: {tau}")
        print(f"o: {o}")
        print(f"Pn: {P_n}")
        return [
            np.array(tau),
            np.array(o),
            np.array(P_n)
        ]

        # return [
        #         np.array(tau).reshape(1, self.N_User).squeeze(),
        #         np.array(o).reshape(1, self.N_User).squeeze(),
        #         np.array(P_n).reshape(1, self.N_User).squeeze()
        #        ]

    def step(self, action):
        self.tau, self.o, self.P_n = self._decomposeAction(action)
        # Environment change
        self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self.User_trajectory + self.U_location
        # state wrap
        state_next = self._wrapState()
        # re-calculate channel gain
        self.ChannelGain = self._ChannelGain_Calculated()
<<<<<<< HEAD

        self.T = self._Time()    # Generate self.T
=======
        self.commonDataRate = self._calculateDataRate(self.H)

        self.T = self._Time()
>>>>>>> f25795235e3fe8f389face313528ef4830898b6d
        # print(reward)
        reward = self.T
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
        self.ChannelGain = self._ChannelGain_Calculated()

        # Generate next state [set of ChannelGain]
        state_next = self._wrapState()
        return state_next

    def close(self):
        pass


if __name__ == '__main__':
    args = get_arguments()
    env = DRGO_env(args)