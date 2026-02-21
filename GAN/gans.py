"""
GAN ile Model MNIST Veri setinden fashion model üretme.
Fashion MNIST 10 class içeren 28x28'lik bir veri setidir.
"""

# import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import FashionMNIST

# Load data;
BUFFER_SIZE = 60000 # Dataset Size.
BATCH_SIZE = 128 # Batch size.
NOISE_DIM = 100 # Noise vector dimension for generator input. 
IMG_SHAPE = (28,28,1) # Input image.
EPOCHS = 2

layers = tf.keras.layers
fashion_mnist = tf.keras.datasets.fashion_mnist

(traing_images,_),(_,_) = fashion_mnist.load_data()
traing_images = traing_images.reshape(-1, 28, 28, 1).astype("float32") # Reshape and convert to float32 for normalization.
traing_images = ((traing_images-127)/-127.5) # Normalization to [-1, 1] range for better training stability.
train_dataset = tf.data.Dataset.from_tensor_slices(traing_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) # Shuffle and batch the dataset.

# Generator model description;
def modelgenerator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias = False, input_shape = (NOISE_DIM, )), # First layer to transform noise vector into a 7x7x256 feature map.
        
        layers.BatchNormalization(), # Increase training stability.
        layers.LeakyReLU(), # Soften negative inputs and allow small gradients to flow.

        layers.Reshape((7,7,256)), # Convert the single dimension to 3D.

        layers.Conv2DTranspose(128,(5,5), strides= (1,1),padding="same",use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(), 
        
        layers.Conv2DTranspose(64,(5,5), strides= (2,2),padding="same",use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(), 

        layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding="same",use_bias=False,activation="tanh"),
        layers.BatchNormalization()
    ])
    return model
#generator=modelgenerator()

# Discriminator description;
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape= IMG_SHAPE),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128,(5,5),strides=(2,2),padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),# Flattens the 3D.
        layers.Dense(1), #Binary classification real/fake.
    ])
    return model

#discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def disciriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output) # Real = 1.
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output) # Fake = 0.
    return real_loss+fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output) # Generator show the real image as fake.

generator = modelgenerator()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
disciriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Helper function Describe;
seed = tf.random.normal([16,NOISE_DIM]) # Constant noise. 

def generate_and_save_images(model,epoch,test_input):
    predictions = model(test_input,training=False) # Just work with generator, not training. We want to see the generated images without updating the model weights.
    
    plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow((predictions[i, :, :, 0]+1)/2, cmap='gray')
        plt.axis('off')

    if not os.path.exists("generate_images"):
        os.makedirs("generate_images")

    plt.savefig(f"generate_images/image_{epoch:03d}.png")
    plt.close()

# Generator ve Discriminator models training;
def train(dataset,epoch):
    for epoch in range(1,epoch+1):
        gen_loss_total = 0 # Generator total loss.
        disc_loss_total = 0 # Discriminator total loss.
        batch_count = 0

        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE,NOISE_DIM]) # Create a Noise.

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise,training=True) # Create a fake image.

                real_output = discriminator(image_batch,training=True) # True image result.
                fake_output = discriminator(generated_images,training=True) # Fake image result.

                gen_loss = generator_loss(fake_output)
                disc_loss = disciriminator_loss(real_output,fake_output)

            gradients_gen = gen_tape.gradient(gen_loss,generator.trainable_variables)
            gradients_disc = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
            disciriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

            gen_loss_total += gen_loss
            disc_loss_total += disc_loss
            batch_count += 1
    print(f"Epoch: {epoch}/{EPOCHS} --- Generator_Loss:{gen_loss_total/batch_count:.3f} --- Discriminator_Loss:{disc_loss_total/batch_count:.3f}")
    generate_and_save_images(generator,epoch,seed) # Save the generated images after each epoch.
                
train(train_dataset,EPOCHS)

 
 
 
 # Finished.