import os
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Input
from undeepvo_model import UnDeepVOModel


def main(args):

    batch_size = 5
    epochs = 100
    # img_rows, img_cols = 128, 416
    img_rows, img_cols = 128, 512
    model_dir = 'models'
    model_name = 'undeepvo'
    model_path = os.path.join(model_dir, model_name)

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = UnDeepVOModel(left_input, right_input)

    model.model.summary()

    plot_model(model.model, show_shapes=True, to_file='scratch/model.png')


if __name__ == '__main__':
    main(None)
