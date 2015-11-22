#!/usr/bin/env python3
import cv2
import numpy as np
import sys

def main():
    filename = sys.argv[1]
    params = np.arange(1, 4)
    imgs = []
    grays = []
    for i, param in enumerate(params):
        imgs.append(cv2.imread(filename))
        grays.append(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY))

        _,thresh = cv2.threshold(grays[i],150,255,cv2.THRESH_BINARY_INV) # threshold
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        dilated = cv2.dilate(thresh,kernel,iterations = param) # dilate
        im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

                # for each contour found, draw a rectangle around it on original image
        for contour in contours:
            [x,y,w,h] = cv2.boundingRect(contour)
            if h < 5 or w < 5: continue
            #if h > grays[i].shape[0] * 0.5 or w > grays[i].shape[1] * 0.5: continue
            cv2.rectangle(imgs[i], (x,y), (x+w,y+h), (255,0,255), 2)

        cv2.imshow('%i' % param,imgs[i])
        print(param)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

main()

