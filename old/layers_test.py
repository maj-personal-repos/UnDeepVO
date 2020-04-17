from layers import spatial_transformation
import matplotlib.image as mpimg
import numpy as np
from keras import backend as K
from image_loader import get_stereo_image_generators
import matplotlib.pyplot as plt

img_rows, img_cols = 128, 512

train_gen, test_gen, train_samples, test_samples = get_stereo_image_generators('data/train/', 'data/test', batch_size=1, shuffle=True)
img = train_gen.__next__()

# spatial transformation lambda layer test

left_image = img[0][0]
disparity = 0.1 * np.ones((1, img_rows, img_cols, 1))

disparity = K.variable(disparity)
left_image = K.variable(left_image)

right_image = spatial_transformation([left_image, disparity], 1, '')

im_right = K.eval(right_image)
plt.imshow(im_right[0])
plt.show()

# TODO: add tests for remaining layers: expand_dims, depth_to_disparity, disparity_difference

