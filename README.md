# **End-to-End Machine Learning Pipeline with DVC and MLflow**

This project demonstrates how to build a robust, end-to-end machine learning pipeline using **DVC (Data Version Control)** for data and model versioning and **MLflow** for experiment tracking. The pipeline is designed to train a **Random Forest Classifier** on the **Pima Indians Diabetes Dataset**, with stages for **data preprocessing**, **model training**, and **evaluation**.


<img width="1362" alt="Screenshot 2025-01-22 at 5 31 34 pm" src="https://github.com/user-attachments/assets/b8fb2394-3e28-4929-98c6-b09f1fbd2f47" />

**Figure**: *Data flow of the end-to-end machine learning pipeline. The diagram illustrates the key stages, including preprocessing, training, and evaluation, and how data moves through the pipeline.*



---

## **Key Features**

### **1. Data Version Control (DVC)**
- Tracks and versions datasets, models, and pipeline stages to ensure reproducibility.
- Automatically re-executes pipeline stages when dependencies change (e.g., data, scripts, parameters).
- Supports remote data storage (e.g., DagsHub, S3) for managing large datasets and models.

### **2. Experiment Tracking with MLflow**
- Logs model hyperparameters (e.g., `n_estimators`, `max_depth`) and performance metrics (e.g., accuracy).
- Tracks experiment metrics, parameters, and artifacts to compare different runs and optimize performance.

---

## **Pipeline Stages**

### **1. Preprocessing**
- The `preprocess.py` script reads the raw dataset (`data/raw/data.csv`), renames columns, and outputs processed data to `data/processed/data.csv`.
- Ensures consistent data preparation across pipeline executions.

### **2. Training**
- The `train.py` script trains a **Random Forest Classifier** on the preprocessed data.
- Saves the trained model as `models/random_forest.pkl`.
- Logs hyperparameters and model artifacts in MLflow for tracking and comparison.

### **3. Evaluation**
- The `evaluate.py` script evaluates the trained model's performance on the dataset.
- Logs evaluation metrics (e.g., accuracy) in MLflow for performance tracking.

---

## **Goals**

### **1. Reproducibility**
- Ensures consistent and reliable workflows by tracking data, parameters, and code with DVC.

### **2. Experimentation**
- Facilitates easy comparison of experiments with different hyperparameters using MLflow.

### **3. Collaboration**
- Enables seamless team collaboration with version control for data, models, and code.

---

## **Use Cases**
- **Data Science Teams**: Track datasets, models, and experiments in an organized and reproducible manner.
- **Machine Learning Research**: Iterate quickly on experiments, track performance, and manage data versions efficiently.

---

## **Technology Stack**
- **Python**: Core language for data processing, model training, and evaluation.
- **DVC**: Tracks version control of data, models, and pipeline stages.
- **MLflow**: Logs and tracks experiments, metrics, and artifacts.
- **Scikit-learn**: Used to build and train the **Random Forest Classifier**.

---

This project demonstrates how to manage the entire lifecycle of a machine learning project by ensuring data, code, models, and experiments are all tracked, versioned, and reproducible.

