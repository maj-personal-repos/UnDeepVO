import tensorflow as tf
import keras
from keras.engine import Layer
from keras.models import Model
from keras.layers import Conv2D, Input, BatchNormalization, Activation, Conv2DTranspose
from keras import backend as K
from tensorflow.contrib.framework import add_arg_scope

batch_size = 100
epochs = 50


@add_arg_scope
def unpool_2d(pool, ind, stride=None, scope='unpool_2d'):
    if stride is None:
        stride = [1, 2, 2, 1]

    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)

        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])

        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret


class MaxPooling2DArgMax(Layer):
    def __init__(self, pool_size=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', **kwargs):
        self.pool_size = pool_size
        self.strides=strides
        self.padding=padding
        super(MaxPooling2DArgMax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaxPooling2DArgMax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        tensor_list = tf.nn.max_pool_with_argmax(inputs, self.pool_size, strides=self.strides, padding=self.padding)
        return [tensor_list[0], tensor_list[1]]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1] / 2, input_shape[2] / 2, input_shape[3]),
                (input_shape[0], input_shape[1] / 2, input_shape[2] / 2, input_shape[3])]


class MaxUnPooling2DArgMax(Layer):
    def __init__(self, stride=(1, 2, 2, 1), **kwargs):
        self.stride = stride
        super(MaxUnPooling2DArgMax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaxUnPooling2DArgMax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        indices = kwargs.get('indices')
        return unpool_2d(inputs, indices)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.stride[1], input_shape[2] * self.stride[2], input_shape[3])


def Conv2DReluBatchNorm(channels, kernel_size, inputs):
    return Activation(activation='relu')(BatchNormalization()(Conv2D(channels, kernel_size=kernel_size)(inputs)))

def DeConv2DReluBatchNorm(channels, kernel_size, inputs):
    return Activation(activation='relu')(BatchNormalization()(Conv2DTranspose(channels, kernel_size=kernel_size)(inputs)))

#
# def Conv2DBlock(channels, kernel_size, inputs):
#     conv1 = Conv2DReluBatchNorm(channels, kernel_size, inputs)
#
#     conv2 = Conv2DReluBatchNorm(channels, kernel_size, conv1)
#
#     return MaxPooling2DArgMax()(conv2)
#
# def DeConv2DBlock(channels, kernel_size, pool, indices):
#
#     unpool1 = MaxUnPooling2DArgMax()(pool, indices=indices)
#
#     unconv1 = Conv2DReluBatchNorm(channels, kernel_size, unpool1)
#
#     return Conv2DReluBatchNorm(channels, kernel_size, unconv1)


# input image dimensions

img_rows, img_cols = 416, 128

# the data


# setup inputs

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


# create input tensor

inputs = Input(input_shape)

conv1 = Conv2DReluBatchNorm(64, (3, 3), inputs)

conv2 = Conv2DReluBatchNorm(64, (3, 3), conv1)

pool1, indices1 = MaxPooling2DArgMax()(conv2)

conv3 = Conv2DReluBatchNorm(128, (3, 3), pool1)

conv4 = Conv2DReluBatchNorm(128, (3, 3), conv3)

pool2, indices2 = MaxPooling2DArgMax()(conv4)

conv5 = Conv2DReluBatchNorm(256, (3, 3), pool2)

conv6 = Conv2DReluBatchNorm(256, (3, 3), conv5)

conv7 = Conv2DReluBatchNorm(256, (1, 1), conv6)

pool3, indices3 = MaxPooling2DArgMax()(conv7)

conv8 = Conv2DReluBatchNorm(512, (3, 3), pool3)

conv9 = Conv2DReluBatchNorm(512, (3, 3), conv8)

conv10 = Conv2DReluBatchNorm(512, (1, 1), conv9)

pool4, indices4 = MaxPooling2DArgMax()(conv10)

#
conv11 = Conv2DReluBatchNorm(512, (3, 3), pool4)

conv12 = Conv2DReluBatchNorm(512, (3, 3), conv11)

conv13 = Conv2DReluBatchNorm(512, (1, 1), conv12)

pool5, indices5 = MaxPooling2DArgMax()(conv13)

unpool5 = MaxUnPooling2DArgMax()(pool5, indices=indices5)

unconv13 = DeConv2DReluBatchNorm(512, (1, 1), unpool5)

unconv12 = DeConv2DReluBatchNorm(512, (3, 3), unconv13)

unconv11 = DeConv2DReluBatchNorm(512, (3, 3), unconv12)
#
unpool4 = MaxUnPooling2DArgMax()(unconv11, indices=indices4)

unconv10 = DeConv2DReluBatchNorm(512, (1, 1), unpool4)

unconv9 = DeConv2DReluBatchNorm(512, (3, 3), unconv10)

unconv8 = DeConv2DReluBatchNorm(512, (3, 3), unconv9)

unpool3 = MaxUnPooling2DArgMax()(unconv8, indices=indices3)

unconv7 = DeConv2DReluBatchNorm(256, (1, 1), unpool3)

unconv6 = DeConv2DReluBatchNorm(256, (3, 3), unconv7)

unconv5 = DeConv2DReluBatchNorm(256, (3, 3), unconv6)

unpool2 = MaxUnPooling2DArgMax()(unconv5, indices=indices2)

unconv4 = DeConv2DReluBatchNorm(128, (3, 3), unpool2)

unconv3 = DeConv2DReluBatchNorm(128, (3, 3), unconv4)

unpool1 = MaxUnPooling2DArgMax()(unconv3, indices=indices1)

unconv2 = DeConv2DReluBatchNorm(64, (3, 3), unpool1)

unconv1 = DeConv2DReluBatchNorm(64, (3, 3), unconv2)

model = Model(inputs=inputs, outputs=unconv1)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

from keras.utils import plot_model

plot_model(model, to_file='model.png')

