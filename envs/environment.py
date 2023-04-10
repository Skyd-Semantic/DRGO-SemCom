import numpy as np
from utils.setting_setup import *
import scipy

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
        self.sigma = 3.9811*(np.e**(-21+7))                        # -174 dBm/Hz -> W/Hz
        # Bandwidth
        self.B = args.bandwidth
        self.B_u = args.bandwidth_u
        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = 1
        self.BS_R_min = 0.1

        """ =============== """
        """     Actions     """
        """ =============== """
        self.o = np.random.randint(0, self.N_User, size=[self.N_User,1])
        # tau is Sub-carrier-Allocation. It is an array with form of Num_Nodes interger number, value change from [0:Num_sub-1] (0 means Sub#1)
        self.tau = np.random.randint(0, self.N_User, size=[self.N_User,1])
        # print(f"{np.shape(self.tau)}-{np.shape(self.tau)}")
        # self.beta = np.reshape(np.random.randint(0, self.N_User, size = self.N_User), self.N_User)
        # eta is AP-Allocation. It is an array with form of Num_Nodes interger number, value change from [0:Num_APs-1] (0 means Sub#1
        self.P_n = np.reshape((np.random.rand(1, self.N_User) * self.P_u_max), (self.N_User,1))
        self.B_u = np.reshape((np.random.rand(1, self.N_User)*self.B), (1, self.N_User))
        """ ========================================= """
        """ ===== Function-based Initialization ===== """
        """ ========================================= """
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        # self.User_trajectory = self._trajectory_U_Generator()
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)

        self.ChannelGain = self._ChannelGain_Calculated()
        self.commonDataRate = self._calculateDataRate(self.ChannelGain)
        self.T = 0                                           # initialize rewards)
        self.ou = 0
        self.contraint1 = 0                                 #intialize contraint1
        self.contraint2 = 0                                 # intialize contraint2
        self.contraint3 = 0                                 # intialize contraint3
        self.contraint4 = 0                                 #intialize contraint4
        """ ============================ """
        """     Environment Settings     """
        """ ============================ """
        self.rewardMatrix = np.array([])
        self.observation_space = self._wrapState().squeeze()
        self.action_space = self._wrapAction()

    def _channelGain_BS_CU(self):
        """     Free-space path loss    """
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

    def _ChannelGain_Calculated(self):
        numerator = self.G_BS_t * self.G_CU_list * (self.lamda ** 2)
        denominator = (4 * np.pi * self.distance_CU_BS) ** 2
        ChannelGain = numerator / denominator
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
        # print(f"Pn: {np.shape(self.P_n)} | H: {np.shape(channelGain_BS_CU)}")
        # print(f"B: {np.shape(self.B)} | Tau: {np.shape(self.tau)} | sigma: {self.sigma}")
        Numerator = ((channelGain_BS_CU))*self.P_n         # self.P must be a list among all users [1, ... , U]
        Denominator = self.B_u * self.tau * self.sigma       # self.B must be a list among all users [1, ... , U]
        # print(f"denominator: {Denominator}")
        # print(f"B: {self.B}")
        # print(f"B_u: {np.shape(self.B_u)}")
        # print(f"P_n: {np.shape(self.P_n)}")
        DataRate = self.B * self.tau * np.log2(1+(Numerator/Denominator))

        # print(f"Numerator: {np.shape(Numerator)} | Denominator: {np.shape(Denominator)} | Datarate: {np.shape(DataRate)}")
        # print(f"======================")
        # print(f"tau: {self.tau}")
        # print(f"======================")
        # print(f"Deno: {self.sigma}")
        # print(f"======================")
        # print(f"Datarate: {DataRate}")
        # print(f"======================")
        return DataRate
    def _Constraint(self):
        # print(f"tau: {self.tau} - tau shape: {np.shape(self.tau)}")
        # print(f"B_u: {self.B_u} - B_u shape: {np.shape(self.B_u)}")
        check = np.zeros((1,self.N_User))
        # print(f"check: {np.shape(check)}")
        for i in range(self.N_User):
            if (self.B_u < 0).any():
                self.contraint1 = 0.03 * self.B_u
            check = np.sum(self.B_u * self.tau) - self.B
            if (check > 0):
                self.contraint2 = check*0.005
                print(f"contrainst2: {self.contraint2}")
            # print(f"check {check}")
            if (self.P_n < 0).any() or ((self.P_n -self.P_u_max)>0).any():
                self.contraint3 = 0.05 * self.P_n
                print(f"contrainst3: {self.contraint3}")
            if (self.o < 0).any() or (self.o > 1).any() :
                self.contraint4 = 0.05 * self.o
                print(f"contrainst4: {self.contraint4}")
    def _Time(self):
        self.DataRate = self._calculateDataRate(self.ChannelGain.reshape(1, -1))
        T = (self.o * self.tau) / self.DataRate
        # print(f"Time: {T}")
        # print(f"Tau: {self.tau}")
        return np.sum(T)
        # return T

    def _wrapState(self):
        self.ChannelGain = self._ChannelGain_Calculated()
        # state = np.concatenate((np.array(self.ChannelGain).reshape(1, -1), np.array(self.U_location).reshape(1, -1),
        #                         np.array(self.User_trajectory).reshape(1, -1)), axis=1)
        state = np.array(self.ChannelGain).reshape(1,-1)
        
        # print(np.shape(state))
        return state

    # def _decomposeState(self, state):
    #     H = state[0: self.N_User]
    #     # print(H)
    #     return [np.array(H)]

    def _wrapAction(self):
        action = np.concatenate((np.array([[self.tau]]).reshape(1, self.N_User),
                                 np.array([[self.o]]).reshape(1, self.N_User),
                                 np.array([[self.P_n]]).reshape(1, self.N_User)), axis=1)

        return action

    def _decomposeAction(self, action):
        # make output for resource allocation tau (range: [0,P_u_max])
        # make output for power (range: [0,1])
        # make output for compression ratio: (range: [0,1])
        tau = action[0][0: self.N_User].astype(float)
        tau = scipy.special.softmax(tau, axis=None)
        # print(f"tau: {tau}")
        o = action[0][self.N_User: 2 * self.N_User].astype(float)
        P_n = (action[0][2 * self.N_User: 3 * self.N_User].astype(float))*self.P_u_max

        # print(f"tau: {tau}")
        # print(f"o: {o}")
        # print(f"P_n: {P_n}")

        return [
            np.array(tau).reshape((1,self.N_User)),
            np.array(o).reshape((1,self.N_User)),
            np.array(P_n).reshape((1,self.N_User))
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

        self.T = self._Time()    # Generate self.T
        A = self._Constraint()
        # print(f"{A}")
        # self.ou = self._sumo()
        # print(f"i {self.ou}")
        # print(reward)
        reward = self.T - self.contraint1 * 0.01 - self.contraint2 * 0.01 - self.contraint3 * 0.01 - self.contraint4 * 0.01

        # reward = T - alpha_1 * constraint_1 - ... - alpha_n * constraint_n
        # T ~ 0.1s -> alpha * constraint < 0.01
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
        self.ChannelGain = self._ChannelGain_Calculated()

        # Generate next state [set of ChannelGain]
        state_next = self._wrapState()
        return state_next

    def close(self):
        pass


if __name__ == '__main__':
    args = get_arguments()
    env = DRGO_env(args)