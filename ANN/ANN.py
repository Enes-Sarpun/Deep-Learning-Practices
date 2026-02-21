"""
MNIST Veri seti:
    Rakamlama 0-9 arası toplamda 10 sınıf var.
    28x28 boyutunda gri tonlamalı görüntüler.
    60,000 eğitim ve 10,000 test görüntüsü içerir.
    Amacımız: ANN ile bu resimleri sınıflandırmak ve tanımlamak.

ANN:
    histogram eşitleme: kontrast eşitleme.
    gaussin blur: gürültü azaltma.
    canny edge detection: kenar tespiti. 

ANN ile MNIST veri sınıflandırma.

Libraries:
    tensorflow: keras ile ANN modeli oluşturmak için.
    numpy: veri işleme.
    matplotlib: veri görselleştirme.
    cv2: OpenCV image processing.
"""

# Import libraries:
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset:
(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load dataset.

print("Eğitim Verisi X_Train Şekli:", x_train.shape) # (60000, 28, 28)
print("Eğitim Verisi Y_Train Şekli:", y_train.shape) # (60000,)

# image preprocessing:
img = x_train[0] # We're getting the first image.
stages = {"Orginal Image":img} 

# Histogram Equalization;
eq = cv2.equalizeHist(img)
stages["Histogram Equalization"] = eq

# Gaussian Blur;
blur = cv2.GaussianBlur(eq, (5, 5), 0)
stages["Gaussian Blur"] = blur

# Canny Edge Detection;
edges = cv2.Canny(blur, 50, 150)
stages["Canny Edge Detection"] =  edges

# Visualizations:
fig,axes = plt.subplots(2,2,figsize=(6,6))
axes = axes.flat
for ax, (title,im) in zip(axes, stages.items()):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("MNIST Image Preprocessing Stages", fontsize=16)
plt.tight_layout()
plt.show()

# Processesing:
def processed_image(img):
    """
    - histogram equalization.
    - gaussin blur.
    - canny edge detection
    - flattering: 28x28 -> 784
    - normalize: 0-255 -> 0-1 
    """

    img_eq = cv2.equalizeHist(img)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    img_edges = cv2.Canny(img_blur,50,150)
    features = img_edges.flatten()/255.0 # 28x28 -> 784
    return features


X_train = np.array([processed_image(img) for img in x_train[:]])
y_train_sub = y_train[:]

X_test = np.array([processed_image(img) for img in x_test[:]])
y_test_sub = y_test[:]

# Ann model creation:
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)), # Input Layer + First Hidden Layer.
    Dropout(0.5), # Layer for reduce overfitting.
    Dense(64, activation='relu'), # second hidden layer.
    Dense(10, activation='softmax') # Output Layer, for 10 classes (0-9)
])

# Compile the model:
model.compile(optimizer=Adam(learning_rate=0.001),loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
# Optimizer, Loss Function and Evaluation Metric.

model.summary() # Model Summary.

# Ann model training:
history = model.fit(
    X_train,y_train_sub,
    validation_data=(X_test,y_test_sub),
    epochs=50,
    batch_size=32,
    verbose=2
)

# Evaluate model perfomance:
test_loss, test_acc = model.evaluate(X_test, y_test_sub)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Plots:
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)

plt.plot(history.history['loss'],label=["Training Loss"])
plt.plot(history.history['val_loss'],label=["Validation Loss"])
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)

plt.plot(history.history['accuracy'],label=["Training Accuracy"])
plt.plot(history.history['val_accuracy'],label=["Validation Accuracy"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

print("Model training completed.")

# Finished.