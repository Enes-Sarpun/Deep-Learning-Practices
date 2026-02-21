"""
flowers dataset:
    rgb: 224x224

CNN ile flower classification ve Problem Solving
"""

# import libraries
from tensorflow_datasets import load
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, # Convolutional layer
    MaxPooling2D, # Pooling layer
    Flatten, # Flatten layer, It reduces multidimensional data to a single dimension.
    Dense, # Fully connected layer, Main layer.
    Dropout # Dropout layer.
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, # Early stopping.
    ReduceLROnPlateau, # Reduce learning rate when a metric has stopped improving.
    ModelCheckpoint, # Save the model after every epoch.
)

import tensorflow as tf
import matplotlib.pyplot as plt

# load dataset

(ds_train, ds_val), ds_info = load(
    'tf_flowers', # flowers dataset.
    split=["train[:80%]", 
           "train[%80:]"], # %80 train, %20 test.
    as_supervised = True, # Labels.
    with_info = True # Information about dataset.
)
print(ds_info.features) # Write the features of dataset.
print(f"Number of classes: {ds_info.features['label'].num_classes} ")

# Example data visualization 
fig = plt.figure(figsize=(10,5))
for i,(image,label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1,3,i+1) # 1 row, 3 columns, i+1 image.
    ax.imshow(image.numpy().astype("uint8")) # Show image.
    ax.set_title(f"Etiket Sayısı: {label.numpy()}") # Add the number of tags as a heading.
    ax.axis("off") # Hide axes.

plt.tight_layout()
plt.show()

# Data augmention + processing
IMG = (180,180)
def preprocess(image,label):
    """ Resize for training, augmention, brightness, contrast, crop, normalize """
    image = tf.image.resize(image,IMG) # 180x180.
    image = tf.image.random_flip_left_right(image) # Random horizontal flip.
    image = tf.image.random_brightness(image,max_delta=0.2) # Brightness adjustment.
    image = tf.image.random_saturation(image,lower=0.9,upper=1.2) # Contrast adjustment.
    image = tf.image.random_crop(image,size=(160,160,3))
    image = tf.image.resize(image,IMG) # 180x180.
    image = tf.cast(image,tf.float32)/255.0 # Normalization.
    return image,label
def val(image,label):
    """ Resize for validation, normalize """
    image = tf.image.resize(image,IMG) # 180x180.
    image = tf.cast(image,tf.float32)/255.0 # Normalization.
    return image,label

# Prepare dataset
ds_train= (
    ds_train.map(preprocess,num_parallel_calls=AUTOTUNE)
    .shuffle(1000) # Mix.
    .batch(32) # Batch size.
    .prefetch(AUTOTUNE) # Load the dataset in advance.
) 

ds_val =(
    ds_val.map(val,num_parallel_calls=AUTOTUNE)
    .batch(32) # Batch size.
    .prefetch(AUTOTUNE) # Load the dataset in advance.
)

# CNN Model Creation;
model = Sequential([
    
    Conv2D(32,(3,3),activation='relu', input_shape = (*IMG,3)), # 32 filtre, 3x3 kernel, relu activation, input shape.
    MaxPooling2D((2,2)), # 2x2 pool size.

    Conv2D(64,(3,3),activation='relu'), # 64 filtre, 3x3 kernel, relu activation.
    MaxPooling2D((2,2)), # 2x2 pool size.

    Conv2D(128,(3,3),activation='relu'), # 128 filtre, 3x3 kernel, relu activation.
    MaxPooling2D((2,2)), # 2x2 pool size.

    # Classification;
    Flatten(), # Flatten layer, It reduces multidimensional data to a single dimension.
    Dense(128,activation='relu'), # 128 nöron, relu activation.
    Dropout(0.5), # %50 dropout.
    Dense(ds_info.features['label'].num_classes,activation='softmax') # Output layer, number of classes, softmax activation.
])

# Callbacks;
callbacks =[
    EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True), # If the validation loss does not improve for 3 epochs, stop training and restore the best weights.
    ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,verbose=1,min_lr=1e-9), # If the validation loss does not improve for 2 epochs, reduce the learning rate by a factor of 0.2.
    ModelCheckpoint('best_model.h5',save_best_only=True) # Save the best model.
]
# Compile the model;
model.compile(
    optimizer=Adam(learning_rate=0.001), # Adam optimizer with a learning rate of 0.001.
    loss='sparse_categorical_crossentropy', # Loss function for multi-class classification.
    metrics=['accuracy'], # Accuracy metric to evaluate the model's performance.
)
# Training; 
history = model.fit(
    ds_train, # Training data.
    validation_data = ds_val, # validation data
    epochs=10, # Number of epochs to train the model.
    callbacks=callbacks, # Callbacks.
    verbose=1, # Show training process.
)

# Model evaluation;
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'],label='Validasyon Doğruluğu') 
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Eğitim Kaybı')
plt.plot(history.history['val_loss'],label='Validasyon Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Finished.