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


def binarize_y_classification(y):
    """
    Transorms dataframes craters positions labels to nupy arrays for classification, 1 if there is crater, 0 ow

    Params
    ------
    y: `panas.DataFrame`
        contains i	row_p	col_p	radius_p information

    Returns
    -------
    y_classification: `numpy array`
        array of the size of the number of images, 1 if there is a crater, 0 ow
    """
    y_classification = y.copy()
    y_classification["has_crater"] = 1
    y_classification = y_classification.drop_duplicates(subset="i")
    y_classification = y_classification[["i", "has_crater"]]
    y_classification = y_classification.set_index("i")
    new_index = list(range(7500))
    y_classification = y_classification.reindex(new_index, fill_value=0)

    return y_classification.values.flatten()
