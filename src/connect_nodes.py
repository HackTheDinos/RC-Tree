#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import tools

# Nodes from corners
def nodes_from_corners(image, points, max_dist=16, iterations=1):
    if iterations == 0:
        return points
    print(iterations)
    nodes = []
    while len(points):
        point = points[-1]
        points.pop()
        neighbors = [point]
        for i in reversed(range(len(points))):
            dist = np.sqrt(np.sum(np.power(np.array(point) - np.array(points[i]), 2)))
            if dist < max_dist:
                neighbors.append(points[i])
                points.pop(i)
        neighbors = np.array(neighbors)
        point = neighbors[np.random.randint(len(neighbors))]
        point = np.mean(neighbors, axis=0)
        print(len(points))
        if iterations == 1:
            #image[point[1], point[0], :] = (255, 0, 0)
            cv2.circle(image, tuple(map(int, point)), 4, (255,0,0), 2)
        nodes.append(point)
    return nodes_from_corners(image, nodes, max_dist + 2, iterations - 1)

# Find the contour of the tree
def find_tree_contour(gray, param):
    _,thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = param) # dilate
    im2, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    out = np.zeros(gray.shape, dtype=np.uint8) + 255
    
    for i, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < gray.shape[0] * 0.5 or w < gray.shape[1] * 0.5: continue
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 1)
        cv2.drawContours(out, contours, i, 0, -1)

    out2 = np.zeros(gray.shape, dtype=np.uint8) + 255
    out2[out == 0] = gray[out == 0]
    #11111out2[out < 2010] = 0
    
    return out2


def find_neighbors(image, points, min_dist=8):
    neighbors = []
    for src_pos in range(len(points)):
        for dst_pos in range(src_pos + 1, len(points)):
            dist = np.sqrt(np.sum(np.power(np.array(point) - np.array(points[i]), 2)))
            if dist > min_dist and is_connected(image, points[src_pos], points[dst_pos]):
                neighbors.append((tuple(points[src_pos]),  tuple(points[dst_pos])))
                break
    return neighbors
        
def main():   
    filename = sys.argv[1]
    param = int(sys.argv[2])
    
    img = cv2.imread(filename)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y,x,:] = np.min(img[y,x])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    tree_contour = find_tree_contour(gray, param)
    cv2.imshow('dst2', tree_contour)
        
    dst = cv2.cornerHarris(tree_contour, 3, 3, 0.01)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    corners = dst > 0.05 * dst.max()
    #img[corners] = [0, 0, 255]

    points = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if corners[y,x]:
                points.append((x,y))

    nodes = nodes_from_corners(img, points, max_dist=6, iterations=3)
    
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
