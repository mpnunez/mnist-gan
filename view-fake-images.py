import numpy as np
from tqdm import tqdm
from datetime import datetime

import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.summary import SummaryWriter 
from keras.models import load_model

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def main():

    #generator = load_model("generator.keras")
    generator = load_model("decoder.keras")

    n_rows = 5
    n_cols = 5
    images_to_view = n_rows * n_cols
    latent_vector_size = 50
    latent_vectors_to_view = np.random.randn(images_to_view,latent_vector_size)
    fake_images = generator(latent_vectors_to_view)

    

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, fake_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()


if __name__ == "__main__":
    main()
