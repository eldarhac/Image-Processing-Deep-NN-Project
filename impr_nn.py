###########################
# Image Processing NN Project
# Neural Networks
# Written by Eldar Hacohen
##########################
import os
import glob
import numpy as np
import keras
from keras import layers, models, optimizers
from keras import backend as K
import random
import pickle
import matplotlib.pyplot as plt
import warnings
from scipy.ndimage.filters import convolve
import impr_nn_utils
import math
from skimage import color
import imageio

CACHE = {}

RESOLUTION = 255
NUM_COLORS = 3


def normalize(imRGB):
    return np.float64(imRGB/RESOLUTION)


def read_image(filename, representation):
    img_array = normalize(imageio.imread(filename))
    if len(img_array.shape) < 3:
        return img_array
    else:
        if representation == 1:
            return color.rgb2gray(img_array)
        else:
            return np.array(img_array, np.float64)


def get_random_patch(im_shape, crop_size):
    """
    :param im_shape: image shape
    :param crop_size: the size to crop as tuple
    :return: indices of random patch in image
    """
    (n_rows, n_cols) = im_shape
    n_rows -= crop_size[0]
    n_cols -= crop_size[1]
    upleft_row = np.random.randint(0, n_rows)
    upleft_col = np.random.randint(0, n_cols)
    return [upleft_row, upleft_col]


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    :param filenames: list of file names
    :param batch_size: the size of each batch
    :param corruption_func:
    :param crop_size:
    :yield: one batch at a time
    """
    while True:
        source_batch = []
        target_batch = []
        (height, width) = crop_size
        for i in range(batch_size):
            fname = random.choice(filenames)
            if fname in CACHE.keys():
                clean_im = CACHE[fname]
            else:
                clean_im = read_image(fname, 1)
                CACHE[fname] = clean_im

            three_rand_patch = get_random_patch(clean_im.shape, (crop_size[0]*3, crop_size[1]*3))
            [t_upleft_row, t_upleft_col] = three_rand_patch
            t_btmdn_row = t_upleft_row + 3*height
            t_btmdn_col = t_upleft_col + 3*width

            three_crop_im = clean_im[t_upleft_row:t_btmdn_row, t_upleft_col:t_btmdn_col]
            corrupt_three_crop = corruption_func(three_crop_im)

            rand_patch = get_random_patch(three_crop_im.shape, crop_size)
            [upleft_row, upleft_col] = rand_patch
            btmdn_row = upleft_row + height
            btmdn_col = upleft_col + width

            clean_patch = three_crop_im[upleft_row:btmdn_row, upleft_col:btmdn_col]
            corrupt_patch = corrupt_three_crop[upleft_row:btmdn_row, upleft_col:btmdn_col]
            source_batch.append(corrupt_patch - 0.5)
            target_batch.append(clean_patch - 0.5)

        source_batch = np.array(source_batch).reshape(batch_size, height, width, 1)
        target_batch = np.array(target_batch).reshape(batch_size, height, width, 1)
        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    :param input_tensor:
    :param num_channels:
    :return: a residual block for the neural net
    """
    X = input_tensor
    first_conv = layers.Conv2D(num_channels, (3, 3), padding="same")
    relu_on_first_conv = layers.Activation('relu')(first_conv(X))
    second_conv = layers.Conv2D(num_channels, (3, 3), padding="same")
    O = second_conv(relu_on_first_conv)
    added = layers.Add()([X, O])
    output = layers.Activation('relu')(added)
    return output


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    :param height:
    :param width:
    :param num_channels:
    :param num_res_blocks:
    :return: builds a model for given parameters
    """
    model_input = layers.Input(shape=(height, width, 1))
    before_res = layers.Conv2D(num_channels, (3, 3), padding="same")
    curr_output = layers.Activation('relu')(before_res(model_input))
    for i in range(num_res_blocks):
        curr_output = resblock(curr_output, num_channels)
    conv_layer = layers.Conv2D(1, (3, 3), padding="same")
    after_res = conv_layer(curr_output)
    model_output = layers.Add()([model_input, after_res])
    return models.Model(inputs=model_input, outputs=model_output)


def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    """
    :param model:
    :param images:
    :param corruption_func:
    :param batch_size:
    :param steps_per_epoch:
    :param num_epochs:
    :param num_valid_samples:
    :return: trains given model according to parameters
    """
    patch_size = (model.input_shape[1], model.input_shape[2])
    n = len(images)
    train_set = images[:int(n*0.8)]
    valid_set = images[int(n*0.8):]
    adam = optimizers.Adam(beta_2=0.9)
    model.compile(adam, loss="mean_squared_error")
    train_generator = load_dataset(train_set, batch_size, corruption_func, patch_size)
    valid_generator = load_dataset(valid_set, batch_size, corruption_func, patch_size)
    x = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, validation_data=valid_generator,
                            validation_steps=num_valid_samples, epochs=num_epochs, use_multiprocessing=True)
    return x.history


def restore_image(corrupted_image, base_model):
    """
    :param corrupted_image:
    :param base_model:
    :return: runs base trained model on given image, fitted to its shape
    """
    a = layers.Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    height, width = corrupted_image.shape
    fixed_model = models.Model(inputs=a, outputs=b)
    corrupted_image = corrupted_image.reshape((1, height, width, 1))
    result = fixed_model.predict(corrupted_image - 0.5) + 0.5
    result_to_return = np.clip(result, 0, 1).astype("float64")
    return result_to_return.reshape((height, width))


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    :param image:
    :param min_sigma:
    :param max_sigma:
    :return: adds random noise to image between given sigmas
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise_to_add = np.random.normal(0, sigma, size=image.shape)
    times255 = np.around((image+noise_to_add) * 255)
    im_to_return = np.clip(np.round(times255) / 255, 0, 1).astype("float64")
    return im_to_return


def add_gaussian_to_use(image):
    """
    wrapper to add gaussian noise
    :param image:
    :return:
    """
    return add_gaussian_noise(image, 0, 0.2)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    learns the model for deblurring
    :param num_res_blocks:
    :param quick_mode:
    :return:
    """
    im_in_batch = 100 if not quick_mode else 10
    steps_per_epoch = 100 if not quick_mode else 3
    num_epochs = 5 if not quick_mode else 2
    num_samples = 1000 if not quick_mode else 30

    list_images = impr_nn_utils.images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)
    history = train_model(model, list_images, add_gaussian_to_use, im_in_batch,
                          steps_per_epoch, num_epochs, num_samples)
    # plt.plot(np.arange(len(history['loss'])) + 1, history['loss'])
    # plt.show()
    # plt.plot(np.arange(len(history['val_loss'])) + 1, history['val_loss'])
    # plt.show()
    return model


def get_filter(filter_size):
    """
    :param filter_size:
    :return: gets a kernel filter to blur with
    """
    n = filter_size - 1
    nom = np.full((1, filter_size), math.factorial(n))
    d_arr = [math.factorial(r)*math.factorial(n - r) for r in range(filter_size)]
    denom = np.array(d_arr)
    filter = (nom / denom).astype(int)
    return filter / np.sum(filter)


def expand(im, filter, add_row=False, add_col=False):
    """
    :param im:
    :param filter:
    :param add_row:
    :param add_col:
    :return: expands image in low res (by adding zeros and blurring)
    """
    from scipy.ndimage.filters import convolve
    [n, m] = im.shape
    padded_im_h = np.insert(im, np.arange(m) + 1, 0, axis=1)
    if add_col:
        padded_im_h = np.append(padded_im_h, np.zeros([n, 1]), axis=1)
    padded_im = np.insert(padded_im_h, np.arange(n) + 1, 0, axis=0)
    if add_row:
        padded_im = np.append(padded_im, np.zeros([1, padded_im_h.shape[1]]), axis=0)
    filtered_im = convolve(2*padded_im, filter)
    filtered_im = convolve(2*filtered_im, filter.T)
    return filtered_im


def add_motion_blur(image, kernel_size, angle):
    """
    :param image:
    :param kernel_size:
    :param angle:
    :return: image with blur and motion
    """
    kernel = impr_nn_utils.motion_blur_kernel(kernel_size, angle)
    blurred_im = convolve(image, kernel)
    return blurred_im


def random_motion_blur(image, list_of_kernel_sizes):
    """
    :param image:
    :param list_of_kernel_sizes:
    :return: image with random motion blur
    """
    from random import choice
    kernel_size = choice(list_of_kernel_sizes)
    angle = np.random.uniform(0, np.pi)
    blurred_im = add_motion_blur(image, kernel_size, angle)
    blurred_im_to_return = np.clip(np.round(blurred_im * 255) / 255, 0, 1).astype("float64")
    return blurred_im_to_return


def add_blur_to_use(image):
    """
    wrapper for add motion blur
    :param image:
    :return:
    """
    return random_motion_blur(image, [7])


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    :param num_res_blocks:
    :param quick_mode:
    :return: learns the deblurring model
    """
    im_in_batch = 100 if not quick_mode else 10
    steps_per_epoch = 100 if not quick_mode else 3
    num_epochs = 5 if not quick_mode else 2
    num_samples = 1000 if not quick_mode else 30

    list_images = impr_nn_utils.images_for_deblurring()
    model = build_nn_model(16, 16, 32, num_res_blocks)
    history = train_model(model, list_images, add_blur_to_use, im_in_batch, steps_per_epoch, num_epochs, num_samples)
    # plt.plot(np.arange(len(history['loss'])) + 1, history['loss'])
    # plt.show()
    # plt.plot(np.arange(len(history['val_loss'])) + 1, history['val_loss'])
    # plt.show()
    return model


def zero_pad_and_blur(im):
    """
    :param im:
    :return: replaces each second row and column by zeros and blurs by 7-kernel
    """
    zeroed_im = np.copy(im)
    a = np.arange(int(im.shape[0] / 2)) * 2
    b = np.arange(int(im.shape[1] / 2)) * 2
    zeroed_im[a] = 0
    zeroed_im[:, b] = 0
    kernel = get_filter(7)
    blurred_im = convolve(zeroed_im*2, kernel)
    blurred_im = convolve(blurred_im*2, kernel.T)
    return blurred_im


def learn_superres_model(num_res_blocks=5, quick_mode=False):
    """
    learns super resolution model
    :param num_res_blocks:
    :param quick_mode:
    :return:
    """
    im_in_batch = 100 if not quick_mode else 10
    steps_per_epoch = 100 if not quick_mode else 3
    num_epochs = 5 if not quick_mode else 2
    num_samples = 1000 if not quick_mode else 30

    list_images = impr_nn_utils.images_for_deblurring()
    model = build_nn_model(64, 64, 128, num_res_blocks)
    history = train_model(model, list_images, zero_pad_and_blur, im_in_batch, steps_per_epoch, num_epochs, num_samples)
    # plt.plot(np.arange(len(history['loss'])) + 1, history['loss'])
    # plt.show()
    # plt.plot(np.arange(len(history['val_loss'])) + 1, history['val_loss'])
    # plt.show()
    return model


def superres_image(im, model):
    """
    creates super resolution version of image
    :param im:
    :param model:
    :return:
    """
    fltr = get_filter(7)
    expand_corrupt_im = expand(im, fltr)
    superres_im = restore_image(expand_corrupt_im, model)
    return superres_im


def train_deep_prior_model(model, corrupted_image, input_im):
    """
    trains deep prior model
    :param model:
    :param corrupted_image:
    :param input_im:
    :return:
    """
    adam = optimizers.Adam(beta_2=0.9)
    model.compile(adam, loss="mean_squared_error")
    model.fit(x=input_im.reshape(1, input_im.shape[0], input_im.shape[1], 1),
              y=corrupted_image.reshape(1, corrupted_image.shape[0],
                                        corrupted_image.shape[1], 1), epochs=1, steps_per_epoch=7200)


def deep_prior_restore_image(corrupted_image):
    """
    *bonus*
    restores image from noise by using the deep prior technique
    :param corrupted_image:
    :return:
    """
    NORM_VARIANCE = 0.5
    if len(corrupted_image.shape) > 2:
        height, width, channels = corrupted_image.shape
    else:
        height, width = corrupted_image.shape
        channels = 1
    model = build_nn_model(height, width, channels, num_res_blocks=10)
    input_im = np.random.uniform(-NORM_VARIANCE, NORM_VARIANCE, corrupted_image.shape)
    train_deep_prior_model(model, corrupted_image - NORM_VARIANCE, input_im)
    return restore_image(corrupted_image, model)


