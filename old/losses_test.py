import numpy as np
from keras import backend as K
from keras.losses import mean_absolute_error

from losses import ssim, photometric_consistency_loss

x = np.ones((1, 10, 10, 1))
y = np.ones((1, 10, 10, 1))

x_img1 = K.variable(x)
y_img1 = K.variable(y)

ssim1 = ssim(x_img1, y_img1)

assert K.eval(ssim1).all() == np.zeros((1, 10, 10)).all()

x_img2 = K.variable(255*x)
y_img2 = K.variable(-255*y)

ssim2 = ssim(x_img2, y_img2)

assert K.eval(ssim2).all() == np.ones((1, 10, 10)).all()

pcl = photometric_consistency_loss(x_img1, y_img1)

assert K.eval(pcl).all() == np.zeros((1, 10, 10)).all()