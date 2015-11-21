from PIL import Image
import numpy as np

TEST_IMG = "images/sample_tree.png"

arr = np.array(Image.open(TEST_IMG).convert('1'))

