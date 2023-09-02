import sys
import numpy as np
from math import log10


def mW2dBm(mW):
    return 10. * log10(mW)


# Function to convert from dBm to mW
def dBm2mW(dBm):
    return 10 ** ((dBm) / 10.)


def dBm2W(dBm):
    return 10 ** (dBm / 10) / 1000


def W2dBm(W):
    # Convert the power vector from watts to dBm
    return 10 * np.log10(W * 1000)
