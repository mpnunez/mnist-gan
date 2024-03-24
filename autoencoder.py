import numpy as np
from tqdm import tqdm
from datetime import datetime

import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy
from tensorflow.summary import SummaryWriter 

def main():

    # Import data
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_mnist = np.concatenate((x_train,x_test))
    x_mnist = x_train.astype("float32") / 255
    x_mnist = np.expand_dims(x_mnist, -1)
    print("x shape:", x_mnist.shape)
    print(len(x_mnist), "MNIST samples")

    LATENT_VECTOR_SIZE = 32
    downsampled_size = (14,14,8)

    # Descriminator network
    image_shape = x_mnist[0].shape
    img_inputs = keras.Input(shape=image_shape)
    x = layers.Flatten()(img_inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    latent_vector = layers.Dense(LATENT_VECTOR_SIZE, activation="linear")(x)
    x = layers.Dense(64, activation="relu")(latent_vector)
    x = layers.Dense(np.prod(image_shape), activation="relu")(x)
    img_outputs = layers.Reshape(image_shape)(x)
    
    encoder = keras.Model(inputs=img_inputs, outputs=latent_vector, name="encoder")
    decoder = keras.Model(inputs=latent_vector, outputs=img_outputs, name="decoder")
    autoencoder = keras.Model(inputs=img_inputs, outputs=img_outputs, name="autoencoder")
    models = [encoder, decoder, autoencoder]
    for model in models:
        model.summary()

    autoencoder.compile(
        loss = "binary_crossentropy",
        optimizer=Adam(),
    )

    N_EPOCHS = 30
    BATCH_SIZE = 64
    autoencoder.fit(x_mnist,x_mnist,batch_size=64, epochs=N_EPOCHS, validation_split=0.2)
    for model in models:
        model.save(f"{model.name}.keras")


if __name__ == "__main__":
    main()
