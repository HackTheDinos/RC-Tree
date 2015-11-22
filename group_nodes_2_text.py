import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from get_nodes import le_main

try:
    import Image
except:
    from PIL import Image

import pytesseract
import os.path
from pytesseract import image_to_string

# %matplotlib inline

def show(img, cvt=cv2.COLOR_GRAY2RGB, do_cvt=True):
    plt.figure(figsize=(10,10))
    if do_cvt: plt.imshow(cv2.cvtColor(img, cvt))
    else: plt.imshow(img)
    plt.show()

imfile = "images/tree_image_ref.png"
#imfile = "images/Tree1.png"
#just_tree = cv2.imread("images/Tree1.png") #use k=12
#just_tree = cv2.imread("images/Tree2.png") #use k=1
just_tree = cv2.imread(imfile) #use k=9 to get only tips, k=4 gets some of the labels but is not as clean

#label_rectangles
#input: png image of the tree, parameter k

#Parameter k: 
#adjust k higher so that phrases are connected, but lower such that all phrases are separate from the tree 

#output: (image, list of rectangles in the format (x,y,w,h))

def label_rectangles(img, thresh=0.5, k=9, iterations=1):
    img = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    #set the amount to dilate to merge words in a phrase together
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k,k))
    #print kernel
    dilated = cv2.dilate(thresh,kernel,iterations)
    im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    #get the rectangle boundaries we want
    heights = []
    rectangles = []
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < 5 or w < 5: continue
        #if h > grays[i].shape[0] * 0.5 or w > grays[i].shape[1] * 0.5: continue
        rectangles.append((x,y,w,h))
        heights.append(h)
    
    #remove the tree rectangle
    #other way: check if square shaped
    good_rectangles = []
    for r in rectangles:
        (x,y,w,h) = r
        if h<3*np.std(heights):
            good_rectangles.append(r)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)

    
    return (img, good_rectangles)


result = label_rectangles(just_tree)    
# show(result[0], False)
# contours = text_contours(just_tree)

# given the rectangles to a text detection program, which hopefully returns ((x,y,w,h),"Genus species")
# also assume we have node locations [(x,y)....]
# now we need to find closest leaf

# labels = [((1,1,10,10), "Genus species")]
labels = [(pos, "Label " + str(result[1].index(pos)) ) for pos in result[1] ]
nodes = le_main(imfile, 1)

associations = []
for label in labels:
    r,lname = label
    x,y,w,h = r
    # get distances to all nodes
    distances = set()
    for node in nodes:
        x1,y1 = node
        d = distance.euclidean((x,y),node)
        distances.add((d,node))
    #get the node that is the minimum distance and name it
    associations.append((min(distances)[1],(x,y),lname))



tree = result[0].copy()
# print list(associations)[0][0]
def print_leaf(leaf):
    nodepos = tuple(map(int, leaf[0]))
    labelpos = tuple(map(int, leaf[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(tree, leaf[2], nodepos, font,0.5, 135)
    cv2.circle(tree, nodepos, 4, (255,0,0), 2)
    cv2.circle(tree, labelpos, 4, (255,0,0),2)
    cv2.line(tree, nodepos, labelpos, (0,0,255), 2)
# map(print_leaf, associations)
# show(tree, False)

def labels_for_tree(just_tree, nodes):
    # just_tree = cv2.imread(image_path) #use k=9 to get only tips, k=4 gets some of the labels but is not as clean
    result = label_rectangles(just_tree)
    labels = [(pos, get_label(Image.fromarray(just_tree), pos) ) for pos in result[1] ]
    associations = []
    for label in labels:
        r,lname = label
        x,y,w,h = r
        # get distances to all nodes
        distances = set()
        for node in nodes:
            x1,y1 = node
            d = distance.euclidean((x,y),node)
            distances.add((d,node))
        #get the node that is the minimum distance and name it
        associations.append((min(distances)[1],(x,y),lname))
    tree = result[0].copy()
    return associations
    # map(print_leaf, associations)
    # show(tree, False)

def imgToStr(im):
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_image = Image.fromarray(gray_im)
    str = pytesseract.image_to_string(gray_image, config="--user-words config/user-words.txt --user-patterns config/user-patterns.txt --tessedit_char_whitelist AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz_0123456789")

    return str



def get_label(img, pos):
    x,y,w,h = pos
    box = (x, y, x + w, y + h)
    ocr_ready = img.crop(box)
    return imgToStr(ocr_ready)
