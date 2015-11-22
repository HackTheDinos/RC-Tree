import numpy as np
import scipy.ndimage
import pickle
import os
import sys
import cv2

np.set_printoptions(linewidth=116)
sys.setrecursionlimit(4096)

def dump_pickle(name, values):
    with open(name, 'wb') as f:
        pickle.dump(values, f)

def load_pickle(name):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def is_connected(image, src, dst):
    return True

