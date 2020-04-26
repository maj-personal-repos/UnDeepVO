from keras import Model
from keras.layers import Conv2D, Conv2DTranspose, concatenate
from keras.optimizers import Adam


class AutoEncoderModel(object):
    def __init__(self, left_input, right_input, lr=1e-4, rows=128, cols=512):
        self.rows = rows
        self.cols = cols
        self.left = left_input
        self.right = right_input
        self.left_est = None
        self.right_est = None
        self.output = None
        self.model = None
        self.lr = lr
        self.build_architecture()
        self.build_outputs()
        self.build_model()

    @staticmethod
    def conv(input, channels, kernel_size, strides, activation='elu'):

        return Conv2D(channels, kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(input)

    @staticmethod
    def deconv(input, channels, kernel_size, scale):

        return Conv2DTranspose(channels, kernel_size=kernel_size, strides=scale, padding='same')(input)

    def conv_block(self, input, channels, kernel_size):
        conv1 = self.conv(input, channels, kernel_size, 1)

        conv2 = self.conv(conv1, channels, kernel_size, 2)

        return conv2

    def deconv_block(self, input, channels, kernel_size, skip):
        deconv1 = self.deconv(input, channels, kernel_size, 2)

        if skip is not None:
            concat1 = concatenate([deconv1, skip], 3)
        else:
            concat1 = deconv1

        iconv1 = self.conv(concat1, channels, kernel_size, 1)

        return iconv1

    def get_output(self, deconv):
        return self.conv(deconv, 3, 3, 1, 'sigmoid')

    def build_architecture(self):
        # encoder
        conv1 = self.conv_block(self.left, 32, 7)
        conv2 = self.conv_block(conv1, 64, 5)
        conv3 = self.conv_block(conv2, 128, 3)
        conv4 = self.conv_block(conv3, 256, 3)
        conv5 = self.conv_block(conv4, 512, 3)
        conv6 = self.conv_block(conv5, 512, 3)
        conv7 = self.conv_block(conv6, 512, 3)

        # skips
        skip1 = conv1
        skip2 = conv2
        skip3 = conv3
        skip4 = conv4
        skip5 = conv5
        skip6 = conv6

        deconv7 = self.deconv_block(conv7, 512, 3, skip6)
        deconv6 = self.deconv_block(deconv7, 512, 3, skip5)
        deconv5 = self.deconv_block(deconv6, 256, 3, skip4)
        deconv4 = self.deconv_block(deconv5, 128, 3, skip3)
        deconv3 = self.deconv_block(deconv4, 64, 3, skip2)
        deconv2 = self.deconv_block(deconv3, 32, 3, skip1)
        deconv1 = self.deconv_block(deconv2, 16, 3, None)

        self.output = self.get_output(deconv1)

    def build_outputs(self):
        self.left_est = self.output
        # self.right_est = expand_dims(self.output, 1, 'right_estimate')

    def build_model(self):
        self.model = Model(inputs=[self.left], outputs=[self.left_est])
        self.model.compile(loss=['mae'],
                           optimizer='adadelta',
                           metrics=['mse'])
