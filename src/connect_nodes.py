#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import tools

def main():   
    
    filename = sys.argv[1]
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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

    nodes = tools.nodes_from_points(img, points)

    #neighbors = tools.find_neighbors(gray, points)
    #print(len(neighbors))
    #for src, dst in neighbors:
    #    cv2.line(img, src, dst, (255,0,0), 1)
    
    
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
