from undeepvo_architecture import get_undeepvo_net
from keras.utils import plot_model

batch_size = 100
epochs = 50

img_rows, img_cols = 256, 832

model = get_undeepvo_net(img_rows, img_cols)

plot_model(model, show_shapes=True, to_file='scratch/model.png')

