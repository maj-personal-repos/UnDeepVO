import tensorflow as tf
from keras.engine import Layer
from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Input, Lambda, concatenate
from keras import backend as K
from keras.models import Model
import keras
from tensorflow.contrib.framework import add_arg_scope


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
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=[input_shape[0], 1, 1, 1])

        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2],
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret


class MaxPooling2DArgMax(Layer):
    def __init__(self, pool_size=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', **kwargs):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
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
        return input_shape[0], input_shape[1] * self.stride[1], input_shape[2] * self.stride[2], input_shape[3]


def conv2d_relu_batchnorm(channels, kernel_size, inputs):
    return Activation(activation='relu')(BatchNormalization()(Conv2D(channels, kernel_size=kernel_size)(inputs)))


def deconv2d_relu_batchnorm(channels, kernel_size, inputs):
    return Activation(activation='relu')(BatchNormalization()(Conv2DTranspose(channels, kernel_size=kernel_size)(inputs)))


def conv2d_block(channels, kernel_size, inputs):
    conv1 = conv2d_relu_batchnorm(channels, kernel_size, inputs)

    conv2 = conv2d_relu_batchnorm(channels, kernel_size, conv1)

    return MaxPooling2DArgMax()(conv2)


def conv2d_block_2(channels, kernel_size, inputs):
    pass


def deconv2d_block(channels, kernel_size, pool, indices):

    unpool1 = MaxUnPooling2DArgMax()(pool, indices=indices)

    unconv1 = deconv2d_relu_batchnorm(channels, kernel_size, unpool1)

    return deconv2d_relu_batchnorm(channels, kernel_size, unconv1)


def deconv2d_block_2(channels, kernel_size, inputs):
    pass


def get_undeepvo_net(img_rows, img_cols, learning_rate=1e-4):

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_rows, 2 * img_cols)
    else:
        input_shape = (img_rows, 2 * img_cols, 3)

    inputs = Input(input_shape)

    left_image = Lambda(lambda x: x[..., :img_cols, :])(inputs)

    right_image = Lambda(lambda x: x[..., img_cols:, :])(inputs)

    cat_image = concatenate([left_image, right_image], axis=3)

    conv1 = conv2d_relu_batchnorm(64, (3, 3), cat_image)

    conv2 = conv2d_relu_batchnorm(64, (3, 3), conv1)

    pool1, indices1 = MaxPooling2DArgMax()(conv2)

    conv3 = conv2d_relu_batchnorm(128, (3, 3), pool1)

    conv4 = conv2d_relu_batchnorm(128, (3, 3), conv3)

    pool2, indices2 = MaxPooling2DArgMax()(conv4)

    conv5 = conv2d_relu_batchnorm(256, (3, 3), pool2)

    conv6 = conv2d_relu_batchnorm(256, (3, 3), conv5)

    conv7 = conv2d_relu_batchnorm(256, (1, 1), conv6)

    pool3, indices3 = MaxPooling2DArgMax()(conv7)

    conv8 = conv2d_relu_batchnorm(512, (3, 3), pool3)

    conv9 = conv2d_relu_batchnorm(512, (3, 3), conv8)

    conv10 = conv2d_relu_batchnorm(512, (1, 1), conv9)

    pool4, indices4 = MaxPooling2DArgMax()(conv10)

    conv11 = conv2d_relu_batchnorm(512, (3, 3), pool4)

    conv12 = conv2d_relu_batchnorm(512, (3, 3), conv11)

    conv13 = conv2d_relu_batchnorm(512, (1, 1), conv12)

    pool5, indices5 = MaxPooling2DArgMax()(conv13)

    unpool5 = MaxUnPooling2DArgMax()(pool5, indices=indices5)

    unconv13 = deconv2d_relu_batchnorm(512, (1, 1), unpool5)

    unconv12 = deconv2d_relu_batchnorm(512, (3, 3), unconv13)

    unconv11 = deconv2d_relu_batchnorm(512, (3, 3), unconv12)

    unpool4 = MaxUnPooling2DArgMax()(unconv11, indices=indices4)

    unconv10 = deconv2d_relu_batchnorm(512, (1, 1), unpool4)

    unconv9 = deconv2d_relu_batchnorm(512, (3, 3), unconv10)

    unconv8 = deconv2d_relu_batchnorm(512, (3, 3), unconv9)

    unpool3 = MaxUnPooling2DArgMax()(unconv8, indices=indices3)

    unconv7 = deconv2d_relu_batchnorm(256, (1, 1), unpool3)

    unconv6 = deconv2d_relu_batchnorm(256, (3, 3), unconv7)

    unconv5 = deconv2d_relu_batchnorm(256, (3, 3), unconv6)

    unpool2 = MaxUnPooling2DArgMax()(unconv5, indices=indices2)

    unconv4 = deconv2d_relu_batchnorm(128, (3, 3), unpool2)

    unconv3 = deconv2d_relu_batchnorm(128, (3, 3), unconv4)

    unpool1 = MaxUnPooling2DArgMax()(unconv3, indices=indices1)

    unconv2 = deconv2d_relu_batchnorm(64, (3, 3), unpool1)

    unconv1 = deconv2d_relu_batchnorm(1, (3, 3), unconv2)

    model = Model(inputs=inputs, outputs=unconv1)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()

    return model

