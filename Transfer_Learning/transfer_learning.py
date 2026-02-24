"""
Pneumonia disease detection with transfer learning.
"""
# Import libraries;
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Image data loading and data augmentation.
from tensorflow.keras.applications import DenseNet121 # Pre-trained model.
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout # Model layers.
from tensorflow.keras.models import Model # Model creation.
from tensorflow.keras.optimizers import Adam # Optimizer.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # Callbacks. 

import matplotlib.pyplot as plt 
import numpy as np
import os
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay # Confusion matrix and visualization.

# Load data;
train_datagen = ImageDataGenerator(
    rescale = 1/255.0, # Normalization to 0-1 range.
    horizontal_flip = True, # Horizontal flip
    rotation_range = 10, # Rotation by +/-10 degrees
    brightness_range = [0.8,1.2], # Brightness adjustment 0.8/1.2
    validation_split = 0.1, # 10% of data split for validation.
) # train_data = train + validation

test_datagen = ImageDataGenerator(rescale=1/255.0)

Data_dir = "chest_xray"
img_size = (224,224)
batch_size  = 64
class_mode = "binary"

train_gen = train_datagen.flow_from_directory(
    os.path.join(Data_dir,"train"), # Folder for training.
    target_size = img_size, # Resize according to img_size.
    batch_size = batch_size, # Batch size.
    class_mode = class_mode, # Binary classification of disease (pneumonia present/absent).
    subset = "training", # Training data.
    shuffle = True, # Shuffling.
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(Data_dir,"train"), # Folder for training.
    target_size = img_size, # Resize according to img_size.
    batch_size = batch_size, # Batch size.
    class_mode = class_mode, # Binary classification of disease (pneumonia present/absent).
    subset = "validation", # Validation data.
    shuffle = False, # Validation series should be ordered.
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(Data_dir,"test"), # Folder for testing.
    target_size = img_size, # Resize according to img_size.
    batch_size = batch_size, # Batch size.
    class_mode = class_mode, # Binary classification of disease (pneumonia present/absent).
    shuffle = False, # Shuffling.
)
# basic visualization
class_name = list(train_gen.class_indices.keys()) # Class names [normal, pneumonia]
images, labels = next(train_gen) # Get a batch (64) of data.

plt.figure(figsize=(10,4))
for i in range(4):
    ax = plt.subplot(1,4,i+1)
    ax.imshow(images[i])
    ax.set_title(class_name[int(labels[i])])
    ax.axis("off")

plt.tight_layout()
plt.show()

# Model describe;
base_model = DenseNet121(
    weights = "imagenet", # We loaded the pre-trained model.
    include_top = False, # Do not include top layers.
    input_shape = (*img_size,3), # Input dimensions (224,224,3).
)
base_model.trainable = False # Freeze the base model, i.e., the base model will not be trained. (Freezes pre-trained model weights for transfer learning.)

x = base_model.output # base model çıktısı.
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation = 'relu')(x) # 128 gizli katman
x = Dropout(0.5)(x)
pred = Dense(1,activation = "sigmoid")(x) # ikili sınıflandırma yapılacağı için 1 katman yeterli.

model = Model(inputs= base_model.input, outputs = pred)

# Model Compile and Callbacks;
model.compile(
    optimizer = Adam(learning_rate=1e-4),
    loss = "binary_crossentropy", # Binary classification
    metrics = ["accuracy"]
)

callbacks = [
    EarlyStopping(monitor = "val_loss",patience = 3, restore_best_weights = True), # Early stopping
    ReduceLROnPlateau(monitor = "val_loss", factor=0.2, patience=2,min_lr=1e-6),
    ModelCheckpoint("bestmodel.h5",monitor="val_loss",save_best_only=True)  
]
print("Model Summary")
print(model.summary()) # Model summary.

# Model training and evaluation
history = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs = 10,
    callbacks = callbacks,
    verbose = 1, # Shows training progress.
)
pred_probs = model.predict(test_gen, verbose = 1)
pred_labels = (pred_probs>0.5).astype(int).ravel() # Generate predictions from probabilities. (If 0.7>0.5 then 1, if 0.3<0.5 then 0).
true_labels = test_gen.classes # True label data.

cm = confusion_matrix(true_labels,pred_labels)
disp = ConfusionMatrixDisplay(cm,display_labels=class_name)

plt.figure(figsize=(8,8))
disp.plot(cmap="Blues",colorbar=False)
plt.title("Test Sets")
plt.show()
plt.savefig("results.png")


# finished.