#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import tools

# Find the contour of the tree
def find_tree_contour(gray, param):
    _,thresh = cv2.threshold(gray, 192, 255, cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = param) # dilate
    im2, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    out = np.zeros(gray.shape, dtype=np.uint8) + 255

    for i, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < gray.shape[0] * 0.5 or w < gray.shape[1] * 0.5: continue
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 1)
        cv2.drawContours(out, contours, i, 0, -1)
    return out

def main():   
    filename = sys.argv[1]
    param = int(sys.argv[2])
    
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    tree_contour = find_tree_contour(gray, param)
    cv2.imshow('dst', tree_contour)
    cv2.imshow('dst2', gray)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return

    _,thresh = cv2.threshold(gray,150,255,0) #cv2.THRESH_BINARY_INV) # threshold 
    gray = np.float32(gray)
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
                points.append((x,y))

    #nodes = tools.nodes_from_points(img, points)
    
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
