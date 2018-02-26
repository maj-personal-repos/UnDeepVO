import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def get_stereo_image_generators(train_folder, img_rows=256, img_cols=832, batch_size=16, shuffle=True):
    train_imagegen = ImageDataGenerator(rescale=1.0 / 255.0,
                                        rotation_range=5,
                                        shear_range=0.01,
                                        zoom_range=0.01,
                                        height_shift_range=0.01,
                                        width_shift_range=0.01)

    train_generator_left = train_imagegen.flow_from_directory(train_folder,
                                                              target_size=(img_rows, img_cols),
                                                              batch_size=batch_size,
                                                              seed=10,
                                                              shuffle=shuffle,
                                                              classes=['image_2'],
                                                              class_mode=None)

    train_generator_right = train_imagegen.flow_from_directory(train_folder,
                                                               target_size=(img_rows, img_cols),
                                                               batch_size=batch_size,
                                                               seed=10,
                                                               shuffle=shuffle,
                                                               classes=['image_3'],
                                                               class_mode=None)

    def train_generator_func():
        while True:
            left_image = train_generator_left.next()
            right_image = train_generator_right.next()

            output = np.concatenate((left_image, right_image), axis=2)

            yield output, [output, np.zeros(shape=(output.shape[0], img_rows - 4, img_cols - 4)),
                               np.zeros(shape=(output.shape[0], img_rows - 4, img_cols - 4))]

    train_generator = train_generator_func()

    return train_generator
