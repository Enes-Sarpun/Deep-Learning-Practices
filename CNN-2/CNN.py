"""
Problem = Kamera kullanılarak ekrana gösterilen sayıların tahmini gerçekleştirilecek.
MNIST Veri seti kullanılacak.
MNIST Veri setinden görseller siyah -> beyaz, ters çevirip kullanacağız. (Beyaz -> Siyah)
255-img(0-255) = image(255-0)
"""

# Import necessary libraries;
from xml.parsers.expat import model
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# load data;
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()

# Images Fixing;
X_Train = 255 - X_Train
X_Test = 255 - X_Test

plt.figure(figsize=(9,3))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(X_Train[i], cmap = 'gray')
    plt.title(f"Label:{y_Train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Normalization and Reshaping;
X_Train = X_Train.reshape(-1,28,28,1).astype('float32') / 255.0
X_Test = X_Test.reshape(-1,28,28,1).astype('float32') / 255.0

# Data Augmentation;
datagen = ImageDataGenerator(
    rotation_range=10,# Random rotation 10 degrees
    zoom_range=0.1,# Zoom in/out %10
    width_shift_range=0.1,# width shift %10
    height_shift_range=0.1# height shift %10
)

# Model Creation;
Model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    #Classification Layers;
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

print(Model.summary())

# Model Compilation;
Model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

# Model Save;
Model.fit(datagen.flow(X_Train,y_Train,batch_size=64),
          epochs=10,
          validation_data=(X_Test,y_Test)
          )
Model.save('mnist_cnn_model.h5')
print("Model Saved.")

# Finished.