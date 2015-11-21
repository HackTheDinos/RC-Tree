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

def find_neighbors(image, points, min_dist = 1.1):
    neighbors = []
    for src_pos in range(len(points)):
        for dst_pos in range(src_pos + 1, len(points)):
            dist = np.sqrt(np.sum(np.power(points[src_pos] - points[dst_pos], 2)))
            if dist > min_dist and is_connected(image, points[src_pos], points[dst_pos]):
                print(points[src_pos], points[dst_pos], dist)
                neighbors.append(points[src_pos] +  points[dst_pos])
                break
    return neighbors
        
