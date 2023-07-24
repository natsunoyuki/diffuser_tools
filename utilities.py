import os
import numpy as np
import matplotlib.pyplot as plt


# %% Plot pipeline outputs.
def plot_images(images, labels = None):
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
    if image_names is None:
        image_names = ["{}.png".format(i) for i in range(len(image_list))]

    assert len(image_list) == len(image_names)

    for i in range(len(image_list)):
        image_list[i].save(os.path.join(save_dir, image_names[i]))