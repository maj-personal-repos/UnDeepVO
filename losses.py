# from keras.layers import Lambda
from keras.losses import mean_absolute_error
import keras.backend as K


def ssim(x, y):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = K.mean(x, axis=-1)
    mu_y = K.mean(y, axis=-1)

    sigma_x = K.mean(x ** 2, axis=-1) - mu_x ** 2
    sigma_y = K.mean(y ** 2, axis=-1) - mu_y ** 2
    sigma_xy = K.mean(x * y, axis=-1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    ssim_out = ssim_n / ssim_d

    return K.clip((1 - ssim_out) / 2, 0, 1)
    # return Lambda(lambda x: K.clip((1-x)/2, 0, 1))(ssim_out)


def photometric_consistency_loss(alpha):
    def loss(y_true, y_pred):
        return alpha * ssim(y_true, y_pred) + (1 - alpha) * mean_absolute_error(y_true, y_pred)

    return loss
