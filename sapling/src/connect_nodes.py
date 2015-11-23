#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from scipy.ndimage import rotate
import operator



def unhook_triangles(nodes, node_neighbors):
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                if (i in node_neighbors[j] and i in node_neighbors[k] and
                    j in node_neighbors[i] and j in node_neighbors[k] and
                    k in node_neighbors[i] and k in node_neighbors[j]):
                    d1 = np.sqrt(np.sum(np.power(np.array(nodes[i]) - np.array(nodes[j]), 2)))
                    d2 = np.sqrt(np.sum(np.power(np.array(nodes[i]) - np.array(nodes[k]), 2)))
                    d3 = np.sqrt(np.sum(np.power(np.array(nodes[j]) - np.array(nodes[k]), 2)))
                    max_d = max(d1, d2, d3)
                    if d1 == max_d:
                        node_neighbors[i] = [x for x in node_neighbors[i] if x != j]
                        node_neighbors[j] = [x for x in node_neighbors[j] if x != i]
                    elif d2 == max_d:
                        node_neighbors[i] = [x for x in node_neighbors[i] if x != k]
                        node_neighbors[k] = [x for x in node_neighbors[k] if x != i]
                    else:
                        node_neighbors[j] = [x for x in node_neighbors[j] if x != k]
                        node_neighbors[k] = [x for x in node_neighbors[k] if x != j]
                        
                    print('Triangle', i, j, k)

def find_root(nodes, node_neighbors):
    leaves = []
    for i in node_neighbors:
        if len(node_neighbors[i]) == 1:
            leaves.append(i)

    max_D = 0
    max_i = 0
    for i in leaves:
        min_d = 1000000
        for j in leaves:
            if i == j:
                continue
            d = np.sqrt(np.sum(np.power(np.array(nodes[i]) - np.array(nodes[j]), 2)))
            if d < min_d:
                min_d = d
        if min_d > max_D:
            max_i = i
            max_D = min_d
    return max_i

def depth_first_cycle_finder(img, nodes, node_neighbors, original_root, root=None, parent=None, sources = {}):
    if root is None:
        sources = {}
        sources[original_root] = original_root
        root = original_root
    for neighbor in node_neighbors[root]:
        if neighbor == parent:
            continue
        if neighbor in sources:
            print("OOPS!", root, neighbor)
            #print(sources)
            path = []
            i = root
            while sources[i] != neighbor:
                path.append(i)
                i = sources[i]
            path.append(i)
            path.append(neighbor)
            #print(path)
            max_path = []
            max_d = 0
            for j in range(len(path)):
                d = np.sqrt(np.sum(np.power(np.array(nodes[path[j]]) - np.array(nodes[path[j-1]]), 2)))
                if d > max_d:
                    max_path = (path[j], path[j - 1])
                    max_d = d
            a, b = max_path
            node_neighbors[a] = [x for x in node_neighbors[a] if x != b]
            node_neighbors[b] = [x for x in node_neighbors[b] if x != a]
            return None
        sources[neighbor] = root
        #print(root, neighbor)
        if depth_first_cycle_finder(img, nodes, node_neighbors, original_root, neighbor, root, sources) == None:
            return None
    return sources
        

def get_rotated_pos(x, y, rot):
    return (int(x * np.cos(rot) - y * np.sin(rot) + 0.5),
            int(x * np.sin(rot) + y * np.cos(rot) + 0.5))

bottom_right = lambda N, M, k: np.fliplr(np.tri(N, M, k=k)) == 1
top_left = lambda N, M, k: np.flipud(np.tri(N, M, k=k)) == 1
bottom_left = lambda N, M, k: np.tri(N, M, k=k) == 1
top_right = lambda N, M, k: np.flipud(np.fliplr(np.tri(N, M, k=k))) == 1

def find_connected_nodes(image, i, nodes):
    out = []
    x, y = nodes[i]
    for j in range(i + 1, len(nodes)):
        if i == j: continue
        distance = np.sqrt(np.sum(np.power(np.array(nodes[i]) - np.array(nodes[j]), 2)))
        dst_x, dst_y = nodes[j]

        angle = np.arctan2(dst_y - y, dst_x - x)
        
        min_x = min(x, dst_x)
        min_y = min(y, dst_y)
        max_x = max(x, dst_x)        
        max_y = max(y, dst_y)
        border = 12
        A = image[min_y - border : max_y + border + 1, min_x - border : max_x + border + 1]
        if np.min(A.shape) == 0:
            continue
        cutout = np.copy(A)
        
        if ((x - min_x) == 0 and (y - min_y) == 0) or ((dst_x - min_x) == 0 and (dst_y - min_y) == 0):
            cutout[top_right(cutout.shape[0], cutout.shape[1], min(cutout.shape) // 2 - cutout.shape[0])] = 255
            cutout[bottom_left(cutout.shape[0], cutout.shape[1], min(cutout.shape) // 2 - cutout.shape[0])] = 255
        else:
            cutout[top_left(cutout.shape[0], cutout.shape[1], min(cutout.shape) // 2 - cutout.shape[0])] = 255
            cutout[bottom_right(cutout.shape[0], cutout.shape[1], min(cutout.shape) // 2 - cutout.shape[0])] = 255
            
        contours = find_tree_contours(cutout, 2)
        for contour in contours:
            other = False
            for k in range(len(nodes)):
                if k == j or k == i: continue
                k_x, k_y = nodes[k]
                k_x = k_x - min_x + border
                k_y = k_y - min_y + border
                if k_x >= border and k_x < cutout.shape[-1] - border and k_y >= border and k_y < cutout.shape[-2] - border:
                    if contour[k_y, k_x] < 200:
                        other = True
                        break

            if other == False and contour[y - min_y + border, x - min_x + border] < 200 and contour[dst_y - min_y + border, dst_x - min_x + border] < 200:
                out.append(j)
    return out

# Nodes from corners
def nodes_from_corners(image, gray, points, max_dist=16, iterations=1):
    if iterations == 0:
        out = []

        for j, point in enumerate(points):
            x, y = map(int, point)
            A = gray[y - 1 : y + 2, x - 1 : x + 2]
            off_y, off_x = np.array(np.unravel_index(A.argmin(), A.shape)) - 1
            new_x, new_y = x + off_x, y + off_y
            if gray[new_y, new_x] < gray[y, x]:
                out.append((new_x,new_y))
            else:
                out.append((x,y))

        return out
    print('iternations:', iterations)
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
        print('Remaining corner points to assign a corner:', len(points))
        if iterations == 1:
            #image[point[1], point[0], :] = (255, 0, 0)
            cv2.circle(image, tuple(map(int, point)), 4, (255,128,0), 2)
        nodes.append(point)
    return nodes_from_corners(image, gray, nodes, max_dist + 2, iterations - 1)


# Find the contour of the tree
def find_tree_contour(gray, param, min_frac=0.5):
    _,thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = param) # dilate
    im2, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    out = np.zeros(gray.shape, dtype=np.uint8) + 255
    
    for i, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if gray.shape[-1] > gray.shape[-2]:
            if w < gray.shape[-1] * min_frac:
                continue
        elif gray.shape[-1] < gray.shape[-2]:
            if h < gray.shape[-2] * min_frac:
                continue
        else:
            if h < gray.shape[-2] * min_frac or w < gray.shape[-1] * min_frac:
                continue
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 1)
        cv2.drawContours(out, contours, i, 0, -1)

    out2 = np.zeros(gray.shape, dtype=np.uint8) + 255
    out2[out == 0] = gray[out == 0]
    return out2

def find_tree_contours(gray, param):
    _,thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    try:
        dilated = cv2.dilate(thresh,kernel,iterations = param) # dilate
    except:
        dilated = cv2.dilate(thresh,kernel,iterations=1) # dilate
    im2, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    
    outs = []
    for i, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < gray.shape[0] * 0.5 or w < gray.shape[1] * 0.5: continue
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 1)
        out = np.zeros(gray.shape, dtype=np.uint8) + 255
        cv2.drawContours(out, contours, i, 0, -1)
        outs.append(out)
    #out2[out < 2010] = 0
    return outs

def main():   
    filename = sys.argv[1]
    param = int(sys.argv[2])
    
    img = cv2.imread(filename)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y,x,:] = np.min(img[y,x])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    tree_contour = find_tree_contour(gray, param)
    cv2.imshow('tree', tree_contour)
        
    dst = cv2.cornerHarris(tree_contour, 3, 3, 0.01)
    dst = cv2.dilate(dst, None)
    corners = dst > 0.05 * dst.max()

    points = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if corners[y,x]:
                points.append((x,y))

    nodes = nodes_from_corners(img, gray, points, max_dist=8, iterations=3)
    for i, node in enumerate(nodes):
        x, y = node

    # Figure out node neighbors
    node_neighbors = {}
    for i, node in enumerate(nodes):
        node_neighbors[i] = find_connected_nodes(gray, i, nodes)

    # Do reciprocal adding of nodes
    for i in range(len(nodes)):
        for j in node_neighbors[i]:
            if i not in node_neighbors[j]:
                node_neighbors[j].append(i)

    # Find Triangles
    unhook_triangles(nodes, node_neighbors)

    # Find Root
    root = find_root(nodes, node_neighbors)
    print("Root:", root)
    
    #sources = breadth_first_disconnect(nodes, node_neighbors, root)
    sources = depth_first_cycle_finder(np.copy(img), nodes, node_neighbors, root)
    while sources is None:
        print("TRYING AGAIN\n")
        sources = depth_first_cycle_finder(np.copy(img), nodes, node_neighbors, root)

    #edges = edges_from_neighbors(node_neighbors)
    #distances = []
    #print(edges)
    #edges = remove_cycles(edges, nodes, root)
    #print(edges)
    #for src, dst in sorted(edges):
    #    cv2.line(img, nodes[src], nodes[dst], (255,0,0), 2)
    
    # Delete nodes with only 2 neighbors as they are connections:
    for i in range(len(nodes)):
        if len(node_neighbors[i]) == 2:
            n1, n2 = node_neighbors[i]
            for j in node_neighbors:
                #print(node_neighbors[j], i, j)
                node_neighbors[j] = [x for x in node_neighbors[j] if x != i]
            node_neighbors[n1].append(n2)
            node_neighbors[n2].append(n1)
            del node_neighbors[i]
    
    # Print lines
    #for i in sorted(sources):
    #    cv2.line(img, nodes[i], nodes[sources[i]], (255,0,0), 2)

    # Print lines
    for i in sorted(node_neighbors):
        node_x, node_y = nodes[i]
        for neighbor in node_neighbors[i]:
            neighbor_x, neighbor_y = nodes[neighbor]
            cv2.line(img, (node_x, node_y), (neighbor_x, neighbor_y), (0,255,0), 3)
            
    #print("SOURCES")
    #for i in sorted(sources):
    #    print(i, sources[i])
                                                                                
    print("NODE NEIGHBORS")
    for i in sorted(node_neighbors):
        print(i, node_neighbors[i])
           
    for i in sorted(node_neighbors):
        cv2.putText(img, str(i), nodes[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    
    tree = {}

        
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
