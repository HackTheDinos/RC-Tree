import cv2
import numpy as np

def find_tree_contour(gray, iterations=2):
    _,thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations) # dilate
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

def get_points(img):
    img = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 4, 3, 0.01)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    corners = dst > 0.05 * dst.max()
    img[corners] = [0, 0, 255]

    points = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if corners[y,x]:
                points.append((y,x))

    return points


def text_contours(img, thresh=0.5, kernel=1, iterations=1):
    img = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations) # dilate
    
#     return dilated
    im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
    
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < 5 or w < 5: continue
        #if h > grays[i].shape[0] * 0.5 or w > grays[i].shape[1] * 0.5: continue
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
        
    return img

def get_labels(points, contours):
    # returns a dict with corresponding labels. e.g.
    # input: [(11,15), (23,55), (12,55)]
    # {(11,15): None, (23,55): Banana, (12,55): None}
    pass
    