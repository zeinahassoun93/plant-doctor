# 🌿 Plant Doctor: AI Disease Diagnosis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/AI-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Prototype-green)]()

### 📝 Overview
**Plant Doctor** is an AI-powered application designed to detect and diagnose plant diseases from leaf images. 

Early detection of crop diseases is critical for food security. This tool uses **Deep Learning (CNNs)** to analyze visual patterns on leaves (spots, blight, rust) and provide an instant diagnosis, helping farmers and gardeners take action before the infection spreads.

### 🧠 The Model
The core of this project is a **Convolutional Neural Network (CNN)** trained on a large dataset of healthy and diseased plant leaves (e.g., PlantVillage dataset).

* **Architecture:** Custom CNN / MobileNetV2 (Transfer Learning).
* **Classes:** Capable of distinguishing between multiple disease classes (e.g., *Tomato Early Blight*, *Potato Late Blight*, *Healthy*).
* **Preprocessing:** Image resizing, normalization, and data augmentation to ensure robust predictions.

### ✨ Key Features
* **📸 Image Analysis:** Upload a photo of a leaf to get an immediate prediction.
* **🔍 Confidence Score:** Returns the probability percentage of the diagnosis.
* **📱 Lightweight:** Designed to be deployable on edge devices or web apps.

### 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python |
| **ML Framework** | TensorFlow / Keras |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib |

---

### 📂 Project Structure
```bash
plant-doctor/
├── models/              # Saved model weights (.h5 / .tflite)
├── src/                 # Source code for training and inference
│   ├── train.py         # Script to train the CNN
│   └── predict.py       # Inference logic
├── app/                 # Application frontend (assets/logo)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

### 🚀 Getting Started
#### 1. Prerequisites
- Python 3.8+ installed.
- pip package manager.

#### 2. Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/zeinahassoun93/plant-doctor.git
cd plant-doctor
pip install -r requirements.txt
```
#### 3. Run Inference
To test the model on a new image:
```bash
python src/predict.py --image path/to/leaf.jpg
```

### 🔮 Future Roadmap
- [ ] Mobile App: Convert the model to TFLite for an Android app (Kotlin).
- [ ] Recommendation Engine: Suggest treatments (fungicides/care tips) based on the diagnosis.
- [ ] Expansion: Add support for more crop types (Wheat, Rice, Corn).
