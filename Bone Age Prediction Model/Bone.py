"""Bone Age Prediction model, Computer Vision Problem."""
# Loading necessry Libraries;
import cv2
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError

# Import Dataset and Cleaning;
df = pd.read_csv("boneage-training-dataset.csv")
image_files = os.path.join("boneage-training-dataset", "boneage-training-dataset")
available_files = set(os.listdir(image_files))
available_ids = set(f.replace(".png","")for f in available_files if f.endswith(".png"))
df = df[df["id"].astype(str).isin(available_ids)].reset_index(drop=True)

# Bone age Normalization;
df["boneage"] = df["boneage"]/240.0
df["path"] = df["id"].apply(lambda x: os.path.join(image_files, f"{x}.png"))
df['male'] = df['male'].astype(int)
scaler = MinMaxScaler()
df['male'] = scaler.fit_transform(df[['male']])

df = df.sample(n=6000, random_state=42).reset_index(drop=True)
print(df.head(5))

# Visualazing;
plt.hist(df["boneage"],bins=50)
plt.title("Bone Age Distribution")
plt.xlabel("Bone Age (Normalized)")
plt.ylabel("Frequency")
#plt.show()

# Reading Images and Resizing;
def load_images(df, target_size=128):
    images=[]
    valid_paths=[]
    for i,path in enumerate(df["path"]):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Image not found:", path)
            continue
        img = cv2.resize(img, (target_size, target_size))
        img = img / 255.0
        images.append(img)
        valid_paths.append(i)
    new_df = df.iloc[valid_paths].reset_index(drop=True)

    X_image = np.array(images).reshape(-1, target_size, target_size,1)
    X_sex = new_df["male"].values.reshape(-1,1)
    y = new_df["boneage"].values

    return X_image, X_sex, y

X_image, X_sex, y = load_images(df)

# Create Test and Train Split;
X_train_img, X_val_img, y_train, y_val, X_train_sex, X_val_sex = train_test_split(
    X_image, y, X_sex, 
    test_size=0.15, random_state=42
)
# Data Augmentation;
datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
datagen.fit(X_train_img)

# Build the CNN Model;
image_input = tf.keras.Input(shape=(128,128,1), name='image_input')
x = Conv2D(32, (3,3), activation='relu')(image_input)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)  
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
cnn_output = Dropout(0.5)(x)

sex_input = tf.keras.Input(shape=(1,), name='sex_input')

combined_features = tf.keras.layers.Concatenate()([cnn_output,sex_input])

z=Dense(64, activation='relu')(combined_features)
z=Dropout(0.5)(z)
output = Dense(1, activation='linear',name="bone_age")(z)

model = tf.keras.Model(inputs=[image_input, sex_input], outputs=[output])

# Compile the Model;
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mae', metrics=[MeanAbsoluteError()])

# Callbacks;
callbacks = [
    EarlyStopping(monitor='val_loss',patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_bone_age_model.keras',save_best_only=True, monitor='val_loss')
]

# Model Training;
history = model.fit(
    [X_train_img, X_train_sex],y_train,
    validation_data=([X_val_img, X_val_sex], y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# Evaluate the Model;
plt.plot(history.history['loss'],label='Train Mae')
plt.plot(history.history['val_loss'],label='Val Mae')
plt.xlabel('epochs')
plt.ylabel('MAE')
plt.title('Model MAE Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

predc = model.predict([X_val_img, X_val_sex])
actuals  = y_val * 240

plt.figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_val_img[i].reshape(128,128),cmap='gray')
    plt.title(f"Pred: {predc[i][0]*240:.1f}\nActual: {actuals[i]:.1f}")
    plt.axis('off')
plt.suptitle("Bone Age Predictions vs Actuals")
plt.tight_layout()
plt.show()

# Final;
final_mae_normalized = history.history['val_mean_absolute_error'][-1]
final_mae_months = final_mae_normalized * 240.0

print(f"\n--- Model Değerlendirmesi ---")
print(f"Normalleştirilmiş Final Val MAE: {final_mae_normalized:.6f}")
print(f"Gerçek Ay Cinsinden Final Val MAE: {final_mae_months:.2f} ay")





# Finished.