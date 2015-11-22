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

def find_neighbors(image, points, min_dist=8):
    neighbors = []
    for src_pos in range(len(points)):
        for dst_pos in range(src_pos + 1, len(points)):
            dist = np.sqrt(np.sum(np.power(np.array(point) - np.array(points[i]), 2)))
            if dist > min_dist and is_connected(image, points[src_pos], points[dst_pos]):
                neighbors.append((tuple(points[src_pos]),  tuple(points[dst_pos])))
                break
    return neighbors
        
def nodes_from_points(image, points, max_dist=16):
    nodes = []
    while len(points):
        weight = 2
        point = points[-1]
        points.pop()
        for i in reversed(range(len(points))):
            dist = np.sqrt(np.sum(np.power(np.array(point) - np.array(points[i]), 2)))
            if dist < max_dist:
                x = points[i][0]
                y = points[i][1]
                point = (int(point[0] * ((weight - 1) / weight) + x * (1 / weight)),
                         int(point[1] * ((weight - 1) / weight) + y * (1 / weight)))
                weight += 1
                points.pop(i)
        for i in reversed(range(len(points))):
            dist = np.sqrt(np.sum(np.power(np.array(point) - np.array(points[i]), 2)))
            if dist < max_dist:
                x = points[i][0]
                y = points[i][1]
                point = (int(point[0] * ((weight - 1) / weight) + x * (1 / weight)),
                         int(point[1] * ((weight - 1) / weight) + y * (1 / weight)))
                weight += 1
                points.pop(i)
        print(point, len(points))
        cv2.circle(image, point, 5, (0,255,255), -1)
