import matplotlib.pyplot as plt
import numpy as np


def show_images(n_images, imgs, cmap=None):
    """
    Randomly displays n_images from a given imgs list

    Params
    ------
    n_images: `int`
        number of images to show
    imgs: `array`
        array of images
    cmap: `String`
    """
    for i in range(n_images):
        idx = np.random.randint(0, len(imgs))
        plt.figure()
        plt.imshow(imgs[idx], cmap=cmap)
        plt.title("Img n° {}".format(idx))
