# uncomment this section to train on the CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from keras.utils import plot_model
from keras.layers import Input
from image_loader import get_stereo_image_generators
from undeepvo_model import UnDeepVOModel
import argparse

parser = argparse.ArgumentParser(description='UnDeepVo Keras implementation.')

parser.add_argument('--mode', type=str, help='train or test', default='train')

parser.add_argument('--models_dir', type=str, help='models directory', default='models')

parser.add_argument('--model_name', type=str, help='model name', default='undeepvo')

parser.add_argument('--data_path', type=str, help='path to the data', required=True)

parser.add_argument('--input_height', type=int, help='input height', default=128)

parser.add_argument('--input_width', type=int, help='input width', default=512)

parser.add_argument('--batch_size', type=int, help='batch size', default=2)

parser.add_argument('--num_epochs', type=int, help='number of epochs', default=50)

parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=0.1)

parser.add_argument('--lr_loss_weight', type=float, help='left-right consistency weight', default=1.0)

parser.add_argument('--alpha_image_loss', type=float, help='weight between SSIM and L1 in the image loss', default=0.85)

parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='log')

parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')

args = parser.parse_args()


def main(args):
    batch_size = args.batch_size

    epochs = args.num_epochs

    img_rows, img_cols = args.input_height, args.input_width

    models_dir = args.models_dir

    model_name = args.model_name

    data_path = args.data_path

    train_gen, test_gen, train_samples, test_samples = get_stereo_image_generators(data_path + '/train',
                                                                                   data_path + '/test',
                                                                                   img_rows=img_rows,
                                                                                   img_cols=img_cols,
                                                                                   batch_size=batch_size,
                                                                                   shuffle=True)

    # channels last by default
    input_shape = (img_rows, img_cols, 3)

    left_input = Input(input_shape)

    left_input_next = Input(input_shape)

    right_input = Input(input_shape)

    udvo = UnDeepVOModel(left_input_next, left_input, right_input, args.learning_rate)

    # for epoch in range(epochs):
    #     # TODO: need to save model after each epoch
    #     # model_path = os.path.join(models_dir, model_name + '_epoch_%d' % epoch)
    #
    #     udvo.model.fit_generator(train_gen,
    #                              steps_per_epoch=train_samples // batch_size,
    #                              epochs=epochs,
    #                              validation_data=test_gen,
    #                              validation_steps=test_samples // batch_size,
    #                              verbose=1,
    #                              # callbacks=[TensorBoard(log_dir=args.log_directory,
    #                              #                        histogram_freq=True,
    #                              #                        batch_size=batch_size,
    #                              #                        write_graph=False,
    #                              #                        write_grads=True)]
    #                              )

    # udvo.model.summary()

    # plot_model(udvo.model, show_shapes=True, to_file='scratch/model.png')


if __name__ == '__main__':
    main(args)
