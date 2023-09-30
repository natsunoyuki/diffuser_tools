import os
import numpy as np
import matplotlib.pyplot as plt


# Utililty functions for visualization, outputs etc.


# %% Plot pipeline outputs.
def plot_images(images, labels = None):
    """Visualizes a set of images generated from the pipeline.
    
    Arguments:
        images: list
            List of RGB images (as arrays) to plot.
        labels: list
            List of corresponding labels of the images.
    """
    N = len(images)
    n_cols = 5
    n_rows = int(np.ceil(N / n_cols))

    plt.figure(figsize = (20, 5 * n_rows))
    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        if labels is not None:
            plt.title(labels[i])
        plt.imshow(np.array(images[i]))
        plt.axis(False)
    plt.show()
    

# %% Save images to disk.
def save_images(image_list, image_names = None, save_dir = "./"):
    """Save images to disk.

    Arguments:
        image_list: list
            List of RGB images (as arrays) to plot.
        image_names: list
            List of corresponding names of the images.
        save_dir: str
            str containing the local directory to output the images to.
    """
    if image_names is None:
        image_names = ["{}.png".format(i) for i in range(len(image_list))]

    assert len(image_list) == len(image_names)

    for i in range(len(image_list)):
        image_list[i].save(os.path.join(save_dir, image_names[i]))