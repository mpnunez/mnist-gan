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

def main():

    # Import data
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_mnist = np.concatenate((x_train,x_test))
    x_mnist = x_train.astype("float32") / 255
    x_mnist = np.expand_dims(x_mnist, -1)
    print("x shape:", x_mnist.shape)
    print(len(x_mnist), "MNIST samples")

    # Descriminator network
    image_shape = x_mnist[0].shape
    descriminator = keras.Sequential(
        [
            keras.Input(shape=image_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    desc_optimizer= Adam(learning_rate=1e-3)
    descriminator.compile(loss="binary_crossentropy", optimizer=desc_optimizer, metrics=["accuracy"])
    print("Descriminator model:")
    descriminator.summary()

    # Generator network
    latent_vector_size = 50
    downsampled_size = (14,14,8)
    generator = keras.Sequential(
        [
            # foundation for 7x7 image
            keras.Input(shape=(latent_vector_size,)),
            layers.Dense(np.prod(downsampled_size), activation="sigmoid"),
            layers.Reshape(downsampled_size),
            # upsample to 28x28
            layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='sigmoid'),
        ]
    )
    print("Generator model:")
    generator.summary()

    # GAN training
    GAN_LEARNING_RATE = 1e-3
    gan_loss_fn = BinaryCrossentropy()
    gan_metric_fn = Accuracy()
    gan_optimizer= Adam(learning_rate=GAN_LEARNING_RATE)

    # Hyperparameters
    n_epochs = 30
    batch_size = 100
    n_real_samples = batch_size // 2
    n_fake_samples = batch_size - n_real_samples
    batches_per_epoch = len(x_mnist) // n_real_samples
    total_batches = n_epochs * batches_per_epoch

    curr_dt = datetime.now()
    timestamp = int(round(curr_dt.timestamp()))
    sw = tf.summary.create_file_writer(f"logdir/logs-{timestamp}")
    images_every_n_batches = 500
    images_per_save = 4
    latent_vectors_to_view = np.random.randn(images_per_save,latent_vector_size)

    for batch_ind in tqdm(range(total_batches)):

        batch_num = batch_ind % batches_per_epoch

        # Real samples to train descriminator
        first_sample = batch_num*n_real_samples
        x_real = x_mnist[first_sample:first_sample+n_real_samples]
        y_real = np.ones(n_real_samples)

        # Fake samples to train descriminator
        latent_vectors = np.random.randn(n_fake_samples,latent_vector_size)
        x_fake = generator(latent_vectors)
        y_fake = np.zeros(n_fake_samples)

        # Combine real and fake samples into batch
        x_real_and_fake = np.concatenate((x_real,x_fake))
        y_real_and_fake = np.concatenate((y_real,y_fake))

        # Train descriminator
        desc_loss, desc_accuracy = descriminator.train_on_batch(x_real_and_fake,y_real_and_fake)
 
        # Back-propagate through full GAN but only update generator
        with tf.GradientTape() as tape:
            desc_fake_predictions = descriminator(generator(latent_vectors))
            desc_fake_predictions = desc_fake_predictions[:,0]
            y_all_real = np.ones(n_fake_samples)
            gan_loss = gan_loss_fn(y_all_real, desc_fake_predictions)
            gan_accuracy = gan_metric_fn(y_all_real, desc_fake_predictions)
 
        grads = tape.gradient(gan_loss, generator.trainable_variables)
        gan_optimizer.apply_gradients(zip(grads, generator.trainable_variables))


        with sw.as_default(step=batch_ind):
            tf.summary.scalar("descriminator-loss", desc_loss)
            tf.summary.scalar("desc-accuracy", desc_accuracy)
            tf.summary.scalar("gan-loss", gan_loss)
            tf.summary.scalar("gan-accuracy", gan_accuracy)
            
            if batch_ind % images_every_n_batches == 0:
                fake_images_to_view = generator(latent_vectors_to_view)
                tf.summary.image("fake images", fake_images_to_view, max_outputs=images_per_save)
                descriminator.save("descriminator.keras")
                generator.save("generator.keras")


if __name__ == "__main__":
    main()
