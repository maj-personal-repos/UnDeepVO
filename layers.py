from keras.layers import Lambda
from util import spatial_transform
import keras.backend as K


def spatial_transformation(inputs, sign, name):
    def output_shape(input_shape):

        return input_shape[0]

    return Lambda(lambda x: spatial_transform(x[0], sign*x[1]), output_shape=output_shape, name=name)(inputs)


def expand_dims(inputs, dimension, name):
    def output_shape(input_shape):
        shape = list(input_shape)

        shape[3] = 1

        return tuple(shape)

    return Lambda(lambda x: K.expand_dims(inputs[:, :, :, dimension], 3), output_shape=output_shape, name=name)(inputs)


def depth_to_disparity(inputs, baseline, focal_length, width, name):
    def output_shape(input_shape):
        return input_shape

    return Lambda(lambda x: width * baseline * focal_length / x, output_shape=output_shape, name=name)(inputs)


def disparity_difference(disparities, name):
    def output_shape(input_shape):
        return input_shape

    return Lambda(lambda x: x[0] - x[1], output_shape=output_shape, name=name)(disparities)
