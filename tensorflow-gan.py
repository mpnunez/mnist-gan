import numpy as np
import keras
from keras import layers
from tqdm import tqdm

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
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    descriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("Descriminator model:")
    descriminator.summary()

    # Generator network
    latent_vector_size = 10
    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_vector_size)),
            layers.Dense(np.prod(image_shape), activation="relu"),
            # layers.Conv2DTranspose(...)
            layers.Reshape(image_shape),
        ]
    )
    print("Generator model:")
    generator.summary()

    # Combined generator + descriminator = GAN
    gan = None

    # Hyperparameters
    n_epochs = 3
    batch_size = 100
    n_real_samples = batch_size // 2
    n_fake_samples = batch_size - n_real_samples
    batches_per_epoch = len(x_mnist) // batch_size
    

    

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        for batch_num in tqdm(range(batches_per_epoch)):

            # Real samples to train descriminator
            first_sample = batch_num*n_real_samples
            x_real = x_mnist[first_sample:first_sample+n_real_samples]
            y_real = np.ones(n_real_samples)

            # Fake samples to train descriminator
            latent_vectors = np.random.random((n_fake_samples,latent_vector_size))
            x_fake = generator(latent_vectors)
            y_fake = np.zeros(n_fake_samples)

            # Combine real and fake samples into batch
            x_real_and_fake = np.concatenate((x_real,x_fake))
            y_real_and_fake = np.concatenate((y_real,y_fake))

            # Train descriminator
            desc_loss = descriminator.train_on_batch(x_real_and_fake,y_real_and_fake)
            #print(f"Descriminator loss: {desc_loss}")

            # Train generator through GAN


    #descriminator.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    #score = descriminator.evaluate(x_test, y_test, verbose=0)


if __name__ == "__main__":
    main()
