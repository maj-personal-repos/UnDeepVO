from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def get_stereo_image_generators(train_folder, test_folder, img_rows=128, img_cols=512, batch_size=16, shuffle=True):
    train_image_gen = ImageDataGenerator(rescale=1.0 / 255.0,
                                         rotation_range=5,
                                         shear_range=0.01,
                                         zoom_range=0.01,
                                         height_shift_range=0.01,
                                         width_shift_range=0.01)

    test_image_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator_left = train_image_gen.flow_from_directory(train_folder,
                                                               target_size=(img_rows, img_cols),
                                                               batch_size=batch_size,
                                                               seed=10,
                                                               shuffle=shuffle,
                                                               classes=['left'],
                                                               class_mode=None,
                                                               follow_links=True)

    train_generator_right = train_image_gen.flow_from_directory(train_folder,
                                                                target_size=(img_rows, img_cols),
                                                                batch_size=batch_size,
                                                                seed=10,
                                                                shuffle=shuffle,
                                                                classes=['right'],
                                                                class_mode=None,
                                                                follow_links=True)

    test_generator_left = test_image_gen.flow_from_directory(test_folder,
                                                             target_size=(img_rows, img_cols),
                                                             batch_size=batch_size,
                                                             seed=10,
                                                             shuffle=shuffle,
                                                             classes=['left'],
                                                             class_mode=None,
                                                             follow_links=True)

    test_generator_right = test_image_gen.flow_from_directory(test_folder,
                                                              target_size=(img_rows, img_cols),
                                                              batch_size=batch_size,
                                                              seed=10,
                                                              shuffle=shuffle,
                                                              classes=['right'],
                                                              class_mode=None,
                                                              follow_links=True)

    def train_generator_func():
        while True:
            left_image = train_generator_left.next()

            right_image = train_generator_right.next()

            yield [left_image, right_image], [left_image,
                                              right_image,
                                              np.zeros(shape=left_image.shape),
                                              np.zeros(shape=right_image.shape)]

    def test_generator_func():
        while True:
            left_image = test_generator_left.next()

            right_image = test_generator_right.next()

            yield [left_image, right_image], [left_image,
                                              right_image,
                                              np.zeros(shape=left_image.shape),
                                              np.zeros(shape=right_image.shape)]

    train_generator = train_generator_func()

    test_generator = test_generator_func()

    train_length = len(train_generator_left.filenames)

    test_length = len(test_generator_left.filenames)

    return train_generator, test_generator, train_length, test_length
