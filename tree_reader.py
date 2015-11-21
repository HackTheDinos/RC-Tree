from PIL import Image
import numpy as np

TEST_IMG = "images/sample_tree.png"

arr = np.array(Image.open(TEST_IMG).convert('1'))

class TreePixel(object):
    def __init__(self, x, y, neighbors):
        self.neighbors = {
            "up"    : 
            "left"  :
            "right" :
            "down"  :
        }

already_checked = set()

for r in xrange(arr.shape[0]):
    for c in xrange(arr.shape[1]):
        if arr[r,c]:
            if (r, c) in already_checked:
                continue
            if arr[r,c]:
                pixels = flood_fill((arr, r,c))
                aleady_checked.update(pixels)
                build_tree(arr, pixels)

def flood_fill(arr, r, c):
    if not arr[r,c]:
        raise Exception("Empty pixel")

    to_fill = set()
    pixels = set()
    
    to_fill.add((r,c))

    while not to_fill.empty():
        (r,c) = to_fill.pop()
        
        if not arr[r,c]:
            continue
        
        pixels.add((r,c))

        to_fill.add((r-1,c))
        to_fill.add((r+1,c))
        to_fill.add((r,c-1))
        to_fill.add((r,c+1))

    return pixels

