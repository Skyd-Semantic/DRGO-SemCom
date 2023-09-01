
import sys
import numpy as np
from math import log10

def mW2dBm(mW):
    return 10. * log10(mW)

# Function to convert from dBm to mW
def dBm2mW(dBm):
    return 10 ** ((dBm) / 10.)

def dBm2W(W):
    return 10 ** (W / 10) / 1000

