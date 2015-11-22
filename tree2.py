import cv2
import numpy as np

def get_points(gray):
    filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 19)
    # filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 1)
    # filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 119, 119)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    dilated = cv2.dilate(filtered,kernel,iterations = 0)

    im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    out = np.zeros(gray.shape, dtype=np.uint8) + 255

    for i, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < gray.shape[0] * 0.04 or w < gray.shape[1] * 0.04: continue

        # must be -1
        cv2.drawContours(out, contours, i, 0, -1)

    corners = cv2.goodFeaturesToTrack(out,400,0.25,8, useHarrisDetector=False)
    corners = np.int0(corners)
    return corners


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


def get_connections(img, points, lines, contours):
    # return all pairs ((x1,y1), (x2,y2)) where points are connected by an edge
    pass

def construct_graph(edges):
    points = set([point for edge in edges for point in edge])
    point_neighbors = {}
    for point in points:
        point_neighbors[point] = set()
        for edge in edges:
            if point in edge:
                point_neighbors[point].update(list(edge))
        point_neighbors[point].discard(point)

    return point_neighbors

def get_leafs(edges, root=None):
    neighbors = construct_graph(edges)
    leafs = set()
    for k, v in neighbors.iteritems():
        if len(v) == 1:
            leafs.add(k)

    leafs.discard(root)
    return leafs

def get_edges(node_neighbors, root=None):
    edges = set()
    for k, v in node_neighbors.iteritems():
        for w in v:
            edges.add(tuple(sorted([k,w])))
    
    to_remove = set()
    if root:
        for edge in edges:
            if root in edge:
                to_remove.add(edge)
    print("to_remove", to_remove)
    print("root",root)
    print("edges",edges)
    print("output",edges - to_remove)
    return edges - to_remove

def build_tree(edges, points_with_labels=None, root=None):
    # print edges
    if points_with_labels == None:
        all_points = set([p for e in edges for p in e])
        points_with_labels = {p:None for p in all_points}
        for leaf in get_leafs(edges, root):
            points_with_labels[leaf] = str(leaf)
        print points_with_labels

    neighbors = construct_graph(edges)
    groups = {}

    processed = set()
    leafs = set([p for p, v in points_with_labels.iteritems() if v is not None])
    groups = {p:v for p, v in points_with_labels.iteritems() if v is not None}

    uh_oh = 0
    while len(groups) > 1:
        uh_oh += 1
        if uh_oh > len(edges):
            import pdb;pdb.set_trace()
            raise Exception("Something is fucked")

        next_leafs = set()
        # print "groups: {}".format(groups)
        # print "processed: {}".format(processed)
        # print "leafs: {}".format(next_leafs)
        for point in leafs:
            if point in processed or point not in leafs:
                continue

            parent = [pt for pt in neighbors[point] if pt not in leafs and pt not in processed]
            print "parent: {}".format(parent)
            assert(len(parent) == 1)

            parent = parent[0]

            if len(neighbors[parent]) == 2:
                if len([pt for pt in neighbors[parent] if pt not in leafs]):
                    next_leafs.add(point)
                    continue

            if len([pt for pt in neighbors[parent] if pt not in leafs]) > 1:
                next_leafs.add(point)
            else:
                group = [pt for pt in neighbors[parent] if pt in leafs]
                groups[parent] = tuple([groups.pop(pt) for pt in group])
                processed.update(group)
                next_leafs.add(parent)

        leafs = next_leafs

    return groups.values()[0]

    # get the lines for each labeled point.
    # get first intersection for that point.
