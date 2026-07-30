# Handwritten Digit Classifier

A production-style Machine Learning project that recognizes handwritten digits (0–9) using a Neural Network implemented with PyTorch. This project demonstrates the complete machine learning workflow, from data loading and preprocessing to model training, evaluation, experiment tracking, and API deployment.

## Project Overview

This project was developed to showcase the skills required for a Junior Machine Learning Engineer role. It follows a clean project structure and includes model training, evaluation, experiment tracking with MLflow, and inference through a FastAPI REST API.

## Features

* Handwritten digit classification (0–9)
* Neural Network implemented with PyTorch
* Data preprocessing and normalization
* Model training and validation
* Performance evaluation using multiple metrics
* Confusion Matrix and Loss Curve visualization
* MLflow experiment tracking
* FastAPI inference API
* Modular and production-style project structure

---

## Tech Stack

* Python
* PyTorch
* NumPy
* SciPy
* Scikit-learn
* Matplotlib
* FastAPI
* Uvicorn
* MLflow
* Git & GitHub

---

## Dataset

The project uses the **ex3data1.mat** dataset from Andrew Ng's Machine Learning course.

Dataset Information

* 5,000 handwritten digit images
* Image size: 20 × 20 pixels
* Flattened input size: 400 features
* Classes: 10 (digits 0–9)

---

## Model Architecture

```
Input Layer (400)

↓

Linear (400 → 25)

↓

Sigmoid Activation

↓

Linear (25 → 10)

↓

Output Logits
```

### Loss Function

CrossEntropyLoss

### Optimizer

Stochastic Gradient Descent (SGD)

### Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Learning Rate | 0.1   |
| Epochs        | 300   |
| Hidden Units  | 25    |

---

## Project Structure

```text
handwritten-digit-classifier/ 
│ 
├── api/ 
├── data/ 
│ └── raw/ 
├── figures/    
├── logs/ 
├── models/ 
├── notebooks/ 
├── src/ 
│   ├── data/ 
│   ├── evaluation/ 
│   ├── inference/ 
    ├── models/ 
│   ├── training/  
│   └── utils/ 
├── tests/
├── train.py 
├── predict.py 
├── requirements.txt 
├── README.md 
└── .gitignore
```

---

## Training

Train the model using:

```bash
python train.py
```

The trained model will be saved in:

```text
models/digit_model.pth
```

---

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Loss Curve

Generated figures are stored in the `figures/` directory.

---

## Experiment Tracking

MLflow is used to track:

* Training loss
* Hyperparameters
* Evaluation metrics
* Generated artifacts
* Model information

---

## API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Open your browser:

* http://127.0.0.1:8000/docs

Available endpoints include:

* `GET /`
* `GET /health`
* `POST /predict`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<Ko-Bhone>/handwritten-digit-classifier.git
```

Move into the project:

```bash
cd handwritten-digit-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Results

Example evaluation outputs include:

* Confusion Matrix
* Loss Curve
* Label Distribution
* Prediction Result
* Average Digit Image

---

## Future Improvements

* Add Convolutional Neural Network (CNN)
* Hyperparameter tuning
* Model versioning
* Docker support
* CI/CD pipeline
* Cloud deployment
* Automated testing

---

## Author

**Bhone Thant Zaw**

Junior Machine Learning Engineer (Bhone Thant Zaw)

GitHub: https://github.com/<Ko-Bhone>

---

## License

This project is for educational and portfolio purposes.
