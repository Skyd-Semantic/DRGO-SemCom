import numpy as np
from utils.setting_setup import *
import scipy

class env_utils:
    def __init__(self):
        pass

    def _channelGain_BS_CU(self):
        """     Free-space path loss    """
        numerator = self.G_BS_t * self.G_CU_list * (self.lamda ** 2)  # Directivity_BS * Directivity_CU * lambda
        denominator = ((4 * np.pi) ** 3) * (self.distance_CU_BS ** 4)
        channelGain = numerator / denominator
        return channelGain

    def _location_BS_Generator(self):
        BS_location = [self.BS_x, self.BS_y]
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

    def _distance_Calculated(self, A, B):
        return np.array([np.sqrt(np.sum((A - B) ** 2, axis=1))]).transpose()

    # def _distance_Calculated(self):
    #       dist = np.zeros((self.Num_BS, self.N_User))
    #       for i in range(self.Num_BS):
    #           for j in range(self.N_User):
    #               dist[i][j] = np.sqrt(np.sum(self.BS_location[i]-self.U_location[j])**2)

    #       return dist

    def _ChannelGain_Calculated(self, sigma_data):
        numerator = self.G_BS_t * self.G_CU_list * (self.lamda ** 2)
        denominator = (4 * np.pi * self.distance_CU_BS) ** 2
        awgn_coeff = np.random.normal(1, sigma_data)
        ChannelGain = (numerator*awgn_coeff) / denominator
        # print(ChannelGain)
        return np.array(ChannelGain)

    def _calculateDataRate(self, channelGain_BS_CU):
        """
        The SNR is measured by:
        :param self:
        :param channelGain_BS_CU:
        :return:
        SNR = Numerator / Denominator
        Numerator = H_k * P_k
        Denominator = N_0 * B_k
        Datarate = B_k np.log2(1+Numerator/Denominator)
        """
        Numerator = channelGain_BS_CU*self.P_n         # self.P must be a list among all users [1, ... , U]
        Denominator = self.B * self.tau * self.naught       # self.B must be a list among all users [1, ... , U]

        DataRate = self.B * self.tau * np.log2(1+(Numerator/Denominator))

        return DataRate, Denominator, Numerator

    def _Time(self):
        self.DataRate, self.deno, self.nume = self._calculateDataRate(self.ChannelGain.reshape(1, -1))
        T = (28000 * self.o) / self.DataRate
        # print(f"Time: {T} - {np.sum(T)}")
        return T

    def _Energy(self):
        E = self.P_n * self.T
        return E

    def _OSigmaMapping(self):
        o_values = [x / (len(self.OSigmaMapping['Comp. Ratio']) - 1)
                    for x in range(len(self.OSigmaMapping['Comp. Ratio']))]
        chosen_comp_loss = []
        chosen_comp_ratio = []
        for o in self.o[0]:
            closest_index = min(range(len(self.OSigmaMapping["Comp. Ratio"])), key=lambda i: abs(o_values[i] - o))
            closest_comp_ratio = self.OSigmaMapping["Comp. Ratio"][closest_index]
            index = self.OSigmaMapping["Comp. Ratio"].index(closest_comp_ratio)
            chosen_comp_ratio.append(closest_comp_ratio)
            chosen_comp_loss.append(self.OSigmaMapping["Loss"][index])
        return np.array([chosen_comp_loss]), np.array([chosen_comp_ratio])