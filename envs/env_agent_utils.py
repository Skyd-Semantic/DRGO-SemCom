import numpy as np
from utils.setting_setup import *
import scipy
from envs.commcal_utils import *

class env_agent_utils:
    def __init__(self):
        pass

    def _wrapState(self):
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        # state = np.array(self.ChannelGain).reshape(1, -1)
        state = np.concatenate((np.array(self.ChannelGain).reshape(1, -1), np.array(self.U_location).reshape(1, -1),
                                np.array(self.User_trajectory).reshape(1, -1)), axis=1)
        return state

    def _decomposeState(self, state):
        H = state[0: self.N_User]
        U_location = state[self.N_User: 2 * self.N_User + 2]
        User_trajectory = state[self.N_User + 2: 2 * self.N_User + 4]
        return [
            np.array(H) , np.array(U_location), np.array(User_trajectory)
        ]

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
        if self.drl_algo == "ddpg-ei":
            tau = scipy.special.softmax(tau, axis=None)
            o = action[0][self.N_User: 2 * self.N_User].astype(float)
            P_n = (action[0][2 * self.N_User: 3 * self.N_User].astype(float)) * self.P_u_max
        else:
            o = 3 * action[0][self.N_User: 2 * self.N_User].astype(float)
            P_n = (action[0][2 * self.N_User: 3 * self.N_User].astype(float)) * 3 * self.P_u_max

        return [
            np.array(tau).reshape((1, self.N_User)),
            np.array(o).reshape((1, self.N_User)),
            np.array(P_n).reshape((1, self.N_User))
        ]

        # return [
        #         np.array(tau).reshape(1, self.N_User).squeeze(),
        #         np.array(o).reshape(1, self.N_User).squeeze(),
        #         np.array(P_n).reshape(1, self.N_User).squeeze()
        #        ]
