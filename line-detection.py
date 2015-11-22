import cv2
import sys
import numpy as np

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 4)
# filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 841, 22)
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

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imwrite('houghlines1.jpg',out)
cv2.imwrite('houghlines2.jpg',img)

    # if h < gray.shape[0] * 0.5 or w < gray.shape[1] * 0.5: continue
    # if h < gray.shape[0] * 0.39 or w < gray.shape[1] * 0.39: continue
    # if h < gray.shape[0] * 0.55 or w < gray.shape[1] * 0.55: continue
    # if h < gray.shape[0] * 0.11 or w < gray.shape[1] * 0.11: continue

# corners = cv2.goodFeaturesToTrack(out,400,0.25,10, useHarrisDetector=False)
# corners = cv2.goodFeaturesToTrack(out,400,0.05,7, useHarrisDetector=True)
# corners = cv2.goodFeaturesToTrack(out,400,0.25,7, useHarrisDetector=True)
# NO HARRIS
# corners = cv2.goodFeaturesToTrack(out,400,0.25,8, useHarrisDetector=False)
# corners = cv2.goodFeaturesToTrack(out,400,0.3,10, useHarrisDetector=False)
# corners = cv2.goodFeaturesToTrack(out,400,0.25,10, useHarrisDetector=True)
# corners = cv2.goodFeaturesToTrack(out,400,0.24,7, useHarrisDetector=False)


    # if h < gray.shape[0] * 0.5 or w < gray.shape[1] * 0.5: continue
    # cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)

# edges = cv2.Canny(gray,50,150,apertureSize = 3)

# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# dilated = cv2.erode(thresh,kernel,iterations = 4) # dilate
# _,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV) # threshold
# _,thresh = cv2.threshold(gray,19,255,cv2.THRESH_BINARY_INV) # threshold
# filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 139, 139)
# graygray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
# graygray = np.float32(graygray)
# dst = cv2.cornerHarris(out, 12, 3, 0.012)

# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()


# edges = cv2.Canny(out,50,150,apertureSize = 3)
# # lines = cv2.HoughLines(out,1,np.pi/180,100)
# lines = cv2.HoughLinesP(edges,3,np.pi/180,100, 80, 55, 0)

# out2 = np.zeros(gray.shape, dtype=np.uint8) + 255

# for l in lines:
#     # for rho,theta in l:
#     for x1,y1,x2,y2 in l:
#         cv2.line(out2,(x1,y1),(x2,y2),(0,255,0),1)
#         # a = np.cos(theta)
#         # b = np.sin(theta)
#         # x0 = a*rho
#         # y0 = b*rho
#         # x1 = int(x0 + 1000*(-b))
#         # y1 = int(y0 + 1000*(a))
#         # x2 = int(x0 - 1000*(-b))
#         # y2 = int(y0 - 1000*(a))

#         # cv2.line(out2,(x1,y1),(x2,y2),(0,0,255),2)






# dst = cv2.cornerHarris(out, 2, 3, 0.13)
# dst = cv2.cornerHarris(out, 3, 11, 0.11)
# dst = cv2.cornerHarris(out, 4, 11, 0.15)
# dst = cv2.cornerHarris(out, 2, 11, 0.08)
# dst = cv2.cornerHarris(out, 2, 11, 0.08)
# dst = cv2.cornerHarris(out, 3, 3, 0.01, borderType=cv2.BORDER_CONSTANT)
# dst = cv2.cornerHarris(out, 2, 11, 0.12, borderType=cv2.BORDER_CONSTANT)

#result is dilated for marking the corners, not important
# dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.11*dst.max()] = [255,0,255]
# img[dst>0.11*dst.max()] = [255,0,255]
# img[dst>0.05*dst.max()] = [255,0,255]
# img[dst>0.01*dst.max()] = [255,0,255]
