from image_loader import get_stereo_image_generators
import matplotlib.pyplot as plt

image_generator = get_stereo_image_generators('data/dataset/sequences/02/', batch_size=1, shuffle=False)

img = image_generator.__next__()
plt.imshow(img[0][0][0, :, :, :])
plt.show()

img = image_generator.__next__()
plt.imshow(img[0][0][0, :, :, :])
plt.show()

img = image_generator.__next__()
plt.imshow(img[0][0][0, :, :, :])
plt.show()

