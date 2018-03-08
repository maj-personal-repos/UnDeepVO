from keras.utils import plot_model
from keras import backend as K
from keras.layers import Input
from undeepvo_model import UnDeepVOModel

batch_size = 100
epochs = 50

# img_rows, img_cols = 128, 416
img_rows, img_cols = 128, 512

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

left_input = Input(input_shape)
right_input = Input(input_shape)

model = UnDeepVOModel(left_input, right_input)

model.model.summary()

plot_model(model.model, show_shapes=True, to_file='scratch/model.png')
