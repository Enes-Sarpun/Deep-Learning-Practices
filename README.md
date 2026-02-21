# 👁️ Computer Vision & Deep Learning

Welcome to my computer vision and deep learning repository!

This repository features a wide range of applications focusing on machine learning, deep learning architectures, data processing, and real-time computer vision tasks.

## 📂 Project Categories

### 1. Object Detection & Tracking (YOLO)
Real-time tracking and detection scripts utilizing **YOLOv8** and **ByteTrack**:
* **`People.py`**: A directional tracking system that detects people and counts how many are entering or exiting a designated area.
* **`Car_Selection.py`**: A multi-class vehicle tracking system (cars, trucks, buses, motorbikes) that counts objects crossing a specific virtual line.
* **`Traffic Plate.py`**: A custom YOLOv8 training pipeline configured for detecting and classifying various traffic signs.

### 2. Human-Computer Interaction & Pose Estimation (MediaPipe)
Applications built with **MediaPipe** for real-time body and face analysis:
* **`MediaPipe.py`**: A fitness tracking script that acts as a Squat Counter by calculating knee joint angles to classify standing, lunging, or squatting poses.
* **`Mediapipe-2.py`**: A facial emotion recognition tool that calculates the distances between facial landmarks to classify emotions (Happy, Surprised, Neutral).
* **`Mediapipe3.py`**: A driver drowsiness detection system that monitors eye aspect ratios and triggers an audio alarm if the driver falls asleep.

### 3. Deep Learning Architectures & Segmentation
Advanced neural network implementations using **TensorFlow/Keras**:
* **`Unet.py`**: Semantic segmentation of satellite imagery using a custom-built U-Net architecture.
* **`gans.py`**: A Generative Adversarial Network (GAN) trained on the Fashion MNIST dataset to synthesize new, artificial clothing images from random noise.
* **`Bone.py`**: A multi-input Convolutional Neural Network (CNN) that predicts bone age from X-ray images by fusing image data with patient gender information.
* **`CNN.py`**: A digit classification model using the MNIST dataset, integrated with robust data augmentation techniques and image inversion preprocessing.
* **`ANN.py`**: An Artificial Neural Network paired with an OpenCV preprocessing pipeline (Histogram Equalization, Gaussian Blur, Canny Edge Detection) for image classification.

### 4. Style Transfer & Vision Transformers
Exploring feature extraction and multi-modal AI:
* **`Still_Transfer.py`**: A Neural Style Transfer script implemented in **PyTorch**. It uses a pre-trained VGG19 model to extract features and apply the artistic style of one image to the content of another.
* **`Image to Describe.py` & `Image To Describe-2.py`**: Image captioning scripts utilizing Hugging Face's **BLIP (Vision Transformer)** model to automatically generate accurate text descriptions from visual inputs.

## 🛠️ Technologies & Libraries Used
* **Deep Learning Frameworks:** PyTorch, TensorFlow, Keras
* **Computer Vision:** OpenCV, MediaPipe, Ultralytics (YOLO)
* **Transformers & NLP:** Hugging Face Transformers (`BlipProcessor`, `BlipForConditionalGeneration`)
* **Data Processing & Visualization:** NumPy, Pandas, Matplotlib, Scikit-Learn

## 🚀 Installation & Usage
# Install dependencies
pip install torch torchvision ultralytics opencv-python mediapipe tensorflow transformers playsound pandas numpy matplotlib scikit-learn


