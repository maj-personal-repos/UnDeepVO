from __future__ import absolute_import, division, print_function
import numpy as np
from math import sin, cos
from keras import backend as K


def euler_to_rotation(theta):
    R_x = np.array([[1, 0, 0],
                    [0, cos(theta[0]), -sin(theta[0])],
                    [0, sin(theta[0]), cos(theta[0])]
                    ])

    R_y = np.array([[cos(theta[1]), 0, sin(theta[1])],
                    [0, 1, 0],
                    [-sin(theta[1]), 0, cos(theta[1])]
                    ])

    R_z = np.array([[cos(theta[2]), -sin(theta[2]), 0],
                    [sin(theta[2]), cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    output = np.identity(4)

    output[:3, :3] = R

    return output


def position_to_translation(position):
    translation_mat = np.identity(4)

    translation_mat[:3, 3] = position[:3]

    return translation_mat


def warp(image, depthmap, pose, K):
    image_shape = K.shape(image)

    num_batch = image_shape[0]

    height = image_shape[1]

    width = image_shape[2]

    channels = image_shape[3]


def spatial_transform(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        rep = K.tile(K.expand_dims(x, 1), [1, n_repeats])

        return rep

    def _interpolate(im, x, y):
        _edge_size = 0

        if _wrap_mode == 'border':
            _edge_size = 1

            im = K.spatial_2d_padding(im, padding=((1, 1), (1, 1)))

            x = x + _edge_size

            y = y + _edge_size

        elif _wrap_mode == 'edge':
            _edge_size = 0

        else:
            return None

        x = K.clip(x, 0.0, K.eval(_width_f) - 1 + 2 * _edge_size)

        x0_f = K.round(x)

        y0_f = K.round(y)

        x1_f = x0_f + 1

        x0 = K.cast(x0_f, 'int32')

        y0 = K.cast(y0_f, 'int32')

        x1 = K.cast(K.minimum(x1_f, K.eval(_width_f) - 1 + 2 * _edge_size), 'int32')

        dim2 = (_width + 2 * _edge_size)

        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)

        base = _repeat(K.arange(_num_batch) * dim1, _height * _width)

        base_y0 = base + y0 * dim2

        idx_l = base_y0 + x0

        idx_r = base_y0 + x1

        im_flat = K.reshape(im, K.stack([-1, _num_channels]))

        pix_l = K.gather(im_flat, idx_l)

        pix_r = K.gather(im_flat, idx_r)

        weight_l = K.expand_dims(x1_f - x, 1)

        weight_r = K.expand_dims(x - x0_f, 1)

        return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        x_t, y_t = np.meshgrid(np.linspace(0.0, K.eval(_width_f) - 1.0, K.eval(_width)),
                               np.linspace(0.0, K.eval(_height_f) - 1.0, K.eval(_height)))

        x_t = K.variable(x_t)

        y_t = K.variable(y_t)

        x_t_flat = K.reshape(x_t, (1, -1))

        y_t_flat = K.reshape(y_t, (1, -1))

        x_t_flat = K.tile(x_t_flat, K.stack([_num_batch, 1]))

        y_t_flat = K.tile(y_t_flat, K.stack([_num_batch, 1]))

        x_t_flat = K.reshape(x_t_flat, [-1])

        y_t_flat = K.reshape(y_t_flat, [-1])

        x_t_flat = x_t_flat + K.reshape(x_offset, [-1]) * _width_f

        input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

        output = K.reshape(input_transformed, K.stack([_num_batch, _height, _width, _num_channels]))

        return output

    _num_batch = K.shape(input_images)[0]

    _height = K.shape(input_images)[1]

    _width = K.shape(input_images)[2]

    _num_channels = K.shape(input_images)[3]

    _height_f = K.cast(_height, 'float32')

    _width_f = K.cast(_width, 'float32')

    _wrap_mode = wrap_mode

    output = _transform(input_images, x_offset)

    return output
