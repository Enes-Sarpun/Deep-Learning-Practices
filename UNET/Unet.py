"""
Satellite image segmentation using UNet.
"""
# import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load dataset;
def load_dataset(root,img_size=(128,128)):
    images, masks = [],[]
    for Tile in sorted(os.listdir(root)): # Iterate through each tile folder in the root directory.
        img_dir = os.path.join(root,Tile,'images') # Directory containing images.
        mask_dir = os.path.join(root,Tile,'masks') # Directory containing masks.
        if not os.path.isdir(img_dir): # If directories don't exist, continue.
            continue
        for f in os.listdir(img_dir): # Iterate through image files.
            if not f.lower().endswith('.jpg'): continue # Only get JPG files.        
            img_path = os.path.join(img_dir,f) # Full path of the image file.
            mask_path = os.path.join(mask_dir,os.path.splitext(f)[0]+".png") # Full path of the mask file.
            if not os.path.exists(mask_path): continue # If mask file doesn't exist, continue.

            # Read and resize image in RGB format.
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # Convert image from BGR to RGB.
            img = cv2.resize(img, img_size)/255.0 # Resize and normalize image.

            # Read mask in grayscale and resize.
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read mask in grayscale.
            mask = cv2.resize(mask, img_size) # Resize mask.
            mask = np.expand_dims(mask, axis=-1)/255.0 # Convert mask to 4D (height, width, channels).

            images.append(img)  # Add image to list.
            masks.append(mask)  # Add mask to list.
    return np.array(images,dtype="float32"), np.array(masks,dtype="float32")  # Convert image and mask lists to numpy arrays and return. 

X,y = load_dataset("Dataset",img_size=(128,128)) # Load dataset.
print(f"Total Images: {len(X)}") # Print total number of images.

X_Train, x_Val, y_Train, y_val = train_test_split(X,y,test_size=0.2) # Split data into training and validation sets.
print(f"Train Images: {len(X_Train)}, Validation Images: {len(x_Val)}") # Print sizes of training and validation sets.

# UNet model;
def unet_model(input_size=(128,128,3)):
    inputs = keras.Input(input_size)

    # Encoder: Feature Extraction and Downsampling
    c1 = layers.Conv2D(16,3,activation="relu",padding="same")(inputs) # First convolution layer.
    c1 = layers.Conv2D(16,3,activation="relu",padding="same")(c1) # Second convolution layer.
    p1 = layers.MaxPooling2D()(c1) # Downsampling with max pooling.
    
    c2 = layers.Conv2D(32,3,activation="relu",padding="same")(p1)
    c2 = layers.Conv2D(32,3,activation="relu",padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64,3,activation="relu",padding="same")(p2)
    c3 = layers.Conv2D(64,3,activation="relu",padding="same")(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(128,3,activation="relu",padding="same")(p3)
    c4 = layers.Conv2D(128,3,activation="relu",padding="same")(c4)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck: Deepest Layer
    c5 = layers.Conv2D(256,3,activation="relu",padding="same")(p4) # Deepest layer.
    c5 = layers.Conv2D(256,3,activation="relu",padding="same")(c5)

    # Decoder: Upsampling and Feature Fusion

    u6 = layers.Conv2DTranspose(128,2,strides=(2,2),padding="same")(c5) # Upsampling layer.
    u6 = layers.concatenate([u6,c4]) # Concatenate features.
    c6 = layers.Conv2D(128,3,activation="relu",padding="same")(u6) # Convolution layer.
    c6 = layers.Conv2D(128,3,activation="relu",padding="same")(c6) 

    u7 = layers.Conv2DTranspose(64,2,strides=(2,2),padding="same")(c6)
    u7 = layers.concatenate([u7,c3]) # Concatenate features.
    c7 = layers.Conv2D(64,3,activation="relu",padding="same")(u7)
    c7 = layers.Conv2D(64,3,activation="relu",padding="same")(c7)

    u8 = layers.Conv2DTranspose(32,2,strides=(2,2),padding="same")(c7)
    u8 = layers.concatenate([u8,c2]) # Concatenate features.
    c8 = layers.Conv2D(32,3,activation="relu",padding="same")(u8)
    c8 = layers.Conv2D(32,3,activation="relu",padding="same")(c8)

    u9 = layers.Conv2DTranspose(16,2,strides=(2,2),padding="same")(c8)
    u9 = layers.concatenate([u9,c1]) # Concatenate features.
    c9 = layers.Conv2D(16,3,activation="relu",padding="same")(u9)
    c9 = layers.Conv2D(16,3,activation="relu",padding="same")(c9)

    outputs = layers.Conv2D(1,1,activation="sigmoid")(c9) # Output layer.
    return keras.Model(inputs, outputs) # Return model.

# Training phase;
unet_model = unet_model() # Create UNet model.
unet_model.compile(optimizer="adam", loss="binary_crossentropy") # Compile model.

# Callbacks;
callbacks = [
    keras.callbacks.ModelCheckpoint("unet_model.h5",save_best_only=True,monitor="val_loss"), # Save best model.
    keras.callbacks.EarlyStopping(patience=10,restore_best_weights = True), # Stop if no improvement for 10 epochs.
    keras.callbacks.ReduceLROnPlateau() # Reduce learning rate.
]

history = unet_model.fit(
    X_Train, y_Train,
    validation_data = (x_Val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Evaluation of results;
plt.plot(history.history['loss'], label='train_loss') # Plot training loss.
plt.plot(history.history['val_loss'], label='val_loss') # Plot validation loss.
plt.legend() # Add legend.
plt.show() # Show plot.

def Show_Predictions(idx=0):
    img = X_Train[idx] # Get image from training set.
    mask = y_Train[idx] # Get mask from training set.
    pred_raw = unet_model.predict(img[None,...])[0].squeeze() # Make prediction with model.
    mask_pred = (pred_raw > 0.5).astype("float32") # Create predicted mask with 0.5 threshold.

    # Results
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.imshow(img) # Show image.
    plt.title("Image") # Add title.
    plt.axis("off") # Turn off axes.

    plt.subplot(1,3,2)
    plt.imshow(mask, cmap="gray") # Show mask in grayscale.
    plt.title("Mask") # Add title.
    plt.axis("off") # Turn off axes.

    plt.subplot(1,3,3)
    plt.imshow(mask_pred, cmap="gray") # Show predicted mask in grayscale.
    plt.title("Prediction") # Add title.
    plt.axis("off") # Turn off axes.
    plt.tight_layout() # Tighten layout.
    plt.show() # Show plot.

Show_Predictions() # Show predictions.

# Finished.