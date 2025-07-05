# IMAGE-CLASSIFICATION-MODEL

**Company:** Codtech It Solutions

**Name:** Lalit Kumar

**Intern Id:** CT4MWZ33

**Domain:** Machine Learning

**Duration:** 16 Weeeks

**Mentor:** Neela Santhosh Kumar
# 🖼️ Image Classification with CNN on CIFAR-10 Dataset

This project implements an image classification pipeline using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. It also includes a Streamlit-based web app where users can upload their own image and see the model's prediction in real-time.

---

## 🌐 Overview

CIFAR-10 is a popular dataset for machine learning and computer vision tasks. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

The main objectives of this project are:

* Build and train a CNN model to classify CIFAR-10 images.
* Evaluate the performance of the model.
* Deploy a web interface using Streamlit to allow user interaction.

## 📊 Dataset Information

* Dataset: CIFAR-10 (loaded via `tensorflow.keras.datasets`)

* Classes:

  * airplane
  * automobile
  * bird
  * cat
  * deer
  * dog
  * frog
  * horse
  * ship
  * truck

* Each image is 32x32 pixels with RGB color.

* The dataset is split into 50,000 training and 10,000 test images.

---

## 👩‍💻 Technologies Used

* Python
* TensorFlow / Keras
* Matplotlib / NumPy
* Streamlit (for the web interface)
* Pillow (image handling)

---

## ⚙️ Model Architecture

The CNN architecture used is simple yet effective:

```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # output layer
])
```

The model uses ReLU activations and the Adam optimizer, with `SparseCategoricalCrossentropy` as the loss function.

---

## 🔄 Training

The model was trained for 10 epochs. Below is the observed performance:

* Final training accuracy: \~81%
* Final test accuracy: \~72.5%

```bash
Test accuracy: 0.7253
```

A training and validation accuracy plot is generated to visualize model performance over time.

---

## 📊 Evaluation Metrics

* Accuracy
* Loss (training and validation)
* Test evaluation using unseen data

Visualization is done with matplotlib to observe learning trends.

---

## 🎨 Streamlit Web App

The app allows users to:

* Upload an image (JPG, PNG, JPEG)
* Preview the uploaded image
* Get a prediction with class label and confidence score

### Example:

Upload a photo of a truck and receive a result like:

```text
Prediction: truck
Confidence: 0.91
```

---

## 📦 Save and Load Model

The model is saved after training using Keras's `model.save()` method:

```python
model.save('cnn_image_classifier.h5')
```

You can reload the model as follows:

```python
model = tf.keras.models.load_model('cnn_image_classifier.h5')
```

> Note: Keras recommends saving in `.keras` format instead of `.h5` for newer workflows.

---

## 🚀 Future Improvements

* Switch to `.keras` model format
* Use data augmentation (e.g., horizontal flip, zoom)
* Apply transfer learning (ResNet, MobileNet)
* Improve UI with Streamlit's advanced widgets
* Allow webcam input or batch image classification
* Deploy the app using Streamlit Cloud or Heroku

---

## 📖 License

This project is licensed under the MIT License.

---

Made with ❤️ using TensorFlow and Streamlit.

# Output
![image](https://github.com/user-attachments/assets/a2178873-c412-4faf-8b3e-8aaf189c46f2)
