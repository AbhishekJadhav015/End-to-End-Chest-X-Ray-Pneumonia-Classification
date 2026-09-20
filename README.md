# 🫁 End-to-End Chest X-Ray Pneumonia Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)

## 📌 Project Overview
An end-to-end computer vision project that classifies chest X-ray images as **Normal** or **Pneumonia**. This project is built with a strict focus on MLOps best practices, featuring a highly modular object-oriented architecture, experiment tracking, and a deployed web interface.

**Live Application:** [Link to your Streamlit Cloud app here]

## 🏗️ System Architecture & Modularity
Unlike standard Jupyter Notebook experiments, this codebase is structured for production environments. It separates data ingestion, model architecture, training orchestrations, and web deployment into independent, scalable components.

```text
pytorch-cnn-project/
├── artifacts/              # Contains production-ready weights (model.pth)
├── src/                    
│   ├── components/         # Core operational modules
│   │   ├── data_ingestion.py   # Dynamic data fetching via kagglehub
│   │   ├── model.py            # ResNet18 Transfer Learning architecture
│   │   └── model_trainer.py    # Training loop with MLflow metric tracking
│   └── pipeline/           # Orchestration layers
│       ├── predict_pipeline.py # Inference logic & tensor transformations
│       └── train_pipeline.py   # Automated training pipeline
├── app.py                  # Streamlit web interface
├── main.py                 # Root execution trigger for retraining
├── requirements.txt        # CPU-optimized dependencies for cloud deployment
└── setup.py                # Package initialization
```

## ⚙️ Key Technical Features
Transfer Learning (PyTorch): Utilized a pre-trained ResNet18 model, froze base convolutional layers to optimize GPU compute, and replaced the fully connected head for binary classification.

Robust Data Pipeline: Implemented custom PyTorch Dataset and DataLoader classes with dynamic os.walk() pathfinding to handle nested Kaggle dataset structures and .png/.jpeg variations.

Experiment Tracking: Integrated MLflow to log hyperparameters, training loss, and validation accuracy across epochs.

Security & Inference: Configured PyTorch's weights_only=True security protocol for loading .pth dictionaries during production inference.

Cloud Deployment: Containerized logic and deployed a lightweight, CPU-optimized web application via Streamlit Cloud.

## 🚀 How to Run Locally
### 1. Clone the repository
```
Bash
git clone [https://github.com/AbhishekJadhav015/End-to-End-Chest-X-Ray-Pneumonia-Classification.git](https://github.com/AbhishekJadhav015/End-to-End-Chest-X-Ray-Pneumonia-Classification.git)
cd pytorch-cnn-project
```
### 2. Set up the environment
Create a virtual environment and install the dependencies:
```
Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Run the Web Application
To test the pre-trained model directly via the UI:
```
Bash
streamlit run app.py
```
### 4. Retrain the Model (Optional)
To trigger the automated data ingestion and training pipeline from scratch:
```
Bash
python main.py
```
📊 Dataset
The model was trained on the Chest X-Ray Images (Pneumonia) dataset, consisting of 5,863 JPEG X-Ray images (anterior-posterior). The pipeline uses kagglehub to fetch this automatically.