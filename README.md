---

# 🍎🥦 Fruit & Vegetable Disease Classification using Deep Learning

A deep learning-based image classification project that detects diseases in fruits and vegetables using transfer learning with pre-trained CNN models like **VGG16** and **ResNet50**.

This project was developed and executed in **Google Colab**, with the dataset stored in Google Drive.

---

## 📌 Project Overview

The goal of this project is to:

* Classify fruits and vegetables as **healthy or diseased**
* Use **transfer learning** for improved accuracy
* Perform dataset splitting (train, validation, test)
* Apply image augmentation techniques
* Train and evaluate a deep learning model
* Save the trained model for future predictions

---

## 📂 Dataset

* Dataset: *Fruit and Vegetable Diseases Dataset*
* Stored in Google Drive
* Contains multiple classes such as:

  * Apple – Healthy
  * Strawberry – Healthy
  * Cucumber – Rotten
  * And other disease categories

The dataset is automatically split into:

* `train`
* `validation`
* `test`

---

## 🛠️ Technologies Used

* Python 3.x
* TensorFlow / Keras
* Scikit-learn
* NumPy
* Google Colab
* Transfer Learning (VGG16 & ResNet50)

---

## 🔄 Data Preprocessing

* Images resized to **224 × 224**
* Pixel values normalized (rescale = 1./255)
* Data augmentation applied:

  * Rotation
  * Width & height shifting
  * Zooming
  * Flipping

---

## 🧠 Model Architecture

The project uses **Transfer Learning**:

### 🔹 VGG16 / ResNet50 (Pre-trained on ImageNet)

* Base model loaded without top layers
* Custom layers added:

  * Flatten layer
  * Dense layers
  * Output layer (Softmax activation)

### ⚙️ Training Configuration

* Optimizer: Adam (learning rate = 0.0001)
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy
* Early Stopping used to prevent overfitting

---

## 📊 Model Training

* Model trained on training dataset
* Validated on validation dataset
* Early stopping monitors `val_accuracy`
* Best model weights restored automatically

---

## 🧪 Testing & Prediction

* Model evaluated on test dataset
* Random images displayed for prediction
* Accuracy and performance analyzed

---

## 💾 Model Saving

The trained model is saved after training for future use and deployment.

---

## 📁 Project Structure

```
Fruit-Disease-Classification/
│
├── Smart_sorting.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/your-repo-name.git
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook

Open in Jupyter Notebook or upload to Google Colab:

```
jupyter notebook
```

---

## 🎯 Key Learning Outcomes

* Understanding Transfer Learning
* Working with ImageDataGenerator
* Handling real-world image datasets
* Preventing overfitting using Early Stopping
* Model evaluation and prediction

---

## 🚀 Future Improvements

* Deploy as a web application
* Convert to mobile app (TensorFlow Lite)
* Add more disease categories
* Improve model accuracy with fine-tuning

---

