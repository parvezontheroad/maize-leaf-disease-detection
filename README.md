# 🌽 Maize Leaf Disease Detection

> **An AI-powered web application for detecting maize leaf diseases using Deep Learning and MobileNetV2.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)](https://flask.palletsprojects.com/)
[![Model](https://img.shields.io/badge/Model-MobileNetV2-green)](https://keras.io/api/applications/mobilenet/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.30%25-success)](#-model-performance)

---

## 🚀 Live Demo

🌐 **[Open the Maize Disease Detection Web App](#)**

> Upload a maize leaf image and the AI model will predict the disease along with its confidence score.

**Note:** The live demo link will be added after permanent deployment.

---

## 📌 About the Project

**Maize Leaf Disease Detection** is an AI-based web application designed to automatically identify common maize leaf diseases from images.

The system uses **Transfer Learning with MobileNetV2**, a lightweight convolutional neural network architecture, to classify maize leaves into four categories.

### 🎯 Disease Classes

| Class | Description |
|---|---|
| 🌿 **Healthy** | Healthy maize leaf |
| 🦠 **Blight** | Maize leaf blight |
| 🍂 **Common Rust** | Common rust infection |
| 🍁 **Gray Leaf Spot** | Gray leaf spot disease |

---

## ✨ Features

- 📷 Upload maize leaf images
- 🤖 AI-powered disease classification
- 🧠 MobileNetV2 deep learning model
- 📊 Prediction confidence score
- ⚡ Fast image processing
- 🌐 Web-based interface
- 📱 Responsive design
- 🔬 Four-class disease classification

---

## 🧠 Model

The project uses **MobileNetV2 with Transfer Learning**.

The ImageNet-pretrained MobileNetV2 feature extractor is combined with custom classification layers for identifying the four maize leaf classes.

### Model Architecture

```text
Input Image
     │
     ▼
224 × 224 × 3
     │
     ▼
MobileNetV2
(ImageNet Pretrained)
     │
     ▼
Global Average Pooling
     │
     ▼
Dropout (0.3)
     │
     ▼
Dense Layer (128)
     │
     ▼
Dropout (0.2)
     │
     ▼
Softmax (4 Classes)
     │
     ▼
Predicted Disease
```

---

## 📊 Model Performance

The final MobileNetV2 model achieved:

| Metric | Result |
|---|---:|
| 🎯 Test Accuracy | **94.30%** |
| 📈 Macro F1-Score | **92.64%** |
| 📊 Weighted F1-Score | **94.25%** |

### Classification Performance

| Disease | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Blight | 91.12% | 89.02% | 90.06% |
| Common Rust | 97.03% | 99.49% | 98.25% |
| Gray Leaf Spot | 83.53% | 81.61% | 82.56% |
| Healthy | 99.43% | 100.00% | 99.72% |

---

## 🗂️ Dataset

The dataset contains **4,188 maize leaf images** distributed across four classes:

- Blight
- Common Rust
- Gray Leaf Spot
- Healthy

The dataset was divided into:

```text
Training     → 2,930 images
Validation   →   626 images
Testing      →   632 images
```

---

## 🛠️ Technologies Used

### Machine Learning
- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow

### Web Development
- Flask
- HTML5
- CSS3
- JavaScript

### Deployment
- Git
- GitHub
- Gunicorn

---

## 📁 Project Structure

```text
Maize-Leaf-Disease-Detection/
│
├── 📄 app.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
│
├── 📁 model/
│   └── MobileNetV2_best.keras
│
└── 📁 templates/
    └── index.html
```

---

## 🔄 How It Works

```text
        Upload Image
             │
             ▼
      Image Preprocessing
             │
             ▼
        MobileNetV2
             │
             ▼
       Disease Prediction
             │
             ▼
    ┌─────────────────────┐
    │ Disease + Confidence│
    └─────────────────────┘
```

---

## 💻 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/parvezontheroad/maize-leaf-disease-detection.git
```

### 2. Open the project directory

```bash
cd maize-leaf-disease-detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

### 5. Open in your browser

```text
http://127.0.0.1:5000
```

---

## 🔮 Future Improvements

- 🌾 Add more diverse real-world field images
- 🧠 Improve model performance on challenging images
- 📍 Add leaf segmentation
- 📱 Develop a mobile application
- ☁️ Deploy the application permanently
- 📊 Add detailed prediction history
- 🌍 Support additional crop diseases

---

## 👨‍💻 Author

**Maize Leaf Disease Detection Project**

Developed as a **Computer Science & Engineering academic project** using Deep Learning and Web Technologies.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Made with ❤️ and Deep Learning 🌽🤖**
