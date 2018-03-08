from keras.optimizers import Adadelta
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, concatenate
from layers import depth_to_disparity, disparity_difference, expand_dims, spatial_transformation
from losses import photometric_consistency_loss


class UnDeepVOModel(object):
    def __init__(self, left_input, right_input, mode='train'):
        # depth = baseline * focal / disparity
        # depth = 0.54 * 721 / (1242 * disp)

        self.baseline = 0.54  # meters

        self.focal_length = 718.856 / 1241  # image width = 1241 (note: must scale using this number)

        self.left = left_input

        self.right = right_input

        self.left_est = None

        self.right_est = None

        self.depthmap = None

        self.depthmap_left = None

        self.depthmap_right = None

        self.disparity_left = None

        self.disparity_right = None

        self.disparity_diff_left = None

        self.disparity_diff_right = None

        self.right_to_left_disparity = None

        self.left_to_right_disparity = None

        self.model_input = None

        self.model = None

        self.depthmap = None

        self.mode = mode

        self.build_architecture()

        self.build_outputs()

        self.build_model()

        if self.mode == 'test':
            return

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

    def get_depth(self, input):
        return self.conv(input, 2, 3, 1, 'sigmoid')

    def build_architecture(self):
        self.model_input = self.left

        # encoder
        conv1 = self.conv_block(self.model_input, 32, 7)

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

        self.depthmap = self.get_depth(deconv1)

    def build_outputs(self):

        # store depthmaps

        self.depthmap_left = expand_dims(self.depthmap, 0)

        self.depthmap_right = expand_dims(self.depthmap, 1)

        if self.mode == 'test':
            return

        # generate disparities

        self.disparity_left = depth_to_disparity(self.depthmap_left, self.baseline, self.focal_length)

        self.disparity_right = depth_to_disparity(self.depthmap_right, self.baseline, self.focal_length)

        # generate estimates of left and right images

        self.left_est = spatial_transformation([self.right, self.disparity_left], -1)

        self.right_est = spatial_transformation([self.left, self.disparity_right], 1)

        # generate left - right consistency

        self.right_to_left_disparity = spatial_transformation([self.disparity_right, self.disparity_left], -1)

        self.left_to_right_disparity = spatial_transformation([self.disparity_left, self.disparity_right], 1)

        self.disparity_diff_left = disparity_difference([self.disparity_left, self.right_to_left_disparity])

        self.disparity_diff_right = disparity_difference([self.disparity_right, self.left_to_right_disparity])

    def build_model(self):
        self.model = Model(inputs=[self.left, self.right], outputs=[self.left_est,
                                                                    self.right_est,
                                                                    self.disparity_diff_left,
                                                                    self.disparity_diff_right])
        self.model.compile(loss=[photometric_consistency_loss,
                                 photometric_consistency_loss,
                                 'mean_absolute_error',
                                 'mean_absolute_error'],
                           optimizer=Adadelta(),
                           metrics=['accuracy'])
