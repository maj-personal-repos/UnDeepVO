import argparse
import os
from keras import Input
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.utils import plot_model
from autoencoder_model import AutoEncoderModel
from autoencoder_image_loader import get_stereo_image_generators
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Stereo Autoencoder')
parser.add_argument('--model_name', type=str, help='model name', default='ae')
parser.add_argument('--models_dir', type=str, help='models directory', default='models')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=128)
parser.add_argument('--input_width', type=int, help='input width', default=512)
parser.add_argument('--batch_size', type=int, help='batch size', default=8)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=100)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-2)
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='log')
args = parser.parse_args()


class VisualizeOutput(Callback):
    def __init__(self, input_image):
        super().__init__()
        self.input_image = input_image

    def on_epoch_begin(self, epoch, logs=None):
        self.visualize_input_output()

    def visualize_input_output(self):
        image = np.concatenate((self.input_image, self.model.predict(self.input_image)), axis=1)[0, :, :, :]
        plt.clf()
        plt.imshow(image)
        plt.pause(1)


def main(args):
    train_gen, test_gen, train_samples, test_samples = get_stereo_image_generators(args.data_path + '/train',
                                                                                   args.data_path + '/test',
                                                                                   img_rows=args.input_height,
                                                                                   img_cols=args.input_width,
                                                                                   batch_size=args.batch_size,
                                                                                   shuffle=False)
    image_generator = get_stereo_image_generators('data/train', 'data/test', batch_size=1, shuffle=False)

    input_image = image_generator[0].__next__()[0][0]

    input_shape = (args.input_height, args.input_width, 3)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    ae = AutoEncoderModel(left_input, right_input, args.learning_rate)
    ae.model.summary()
    plot_model(ae.model, show_shapes=True, to_file='scratch/ae.png')
    ae.model.fit_generator(train_gen,
                           steps_per_epoch=train_samples // args.batch_size,
                           # validation_data=test_gen,
                           # validation_steps=test_samples // args.batch_size,
                           epochs=args.num_epochs,
                           verbose=1,
                           callbacks=[VisualizeOutput(input_image),
                                      TensorBoard(log_dir=args.log_directory,
                                                  batch_size=args.batch_size,
                                                  write_graph=False),
                                      ModelCheckpoint(os.path.join(args.models_dir, args.model_name + '.h5'),
                                                      monitor='loss',
                                                      verbose=1,
                                                      save_best_only=True)])


if __name__ == '__main__':
    main(args)
