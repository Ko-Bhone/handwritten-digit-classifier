# Handwritten Digit Classifier

A production-style Machine Learning project that recognizes handwritten digits (0–9) using a Neural Network implemented with PyTorch. This project demonstrates the complete machine learning workflow, from data loading and preprocessing to model training, evaluation, experiment tracking, and API deployment.

## Project Overview

This project was developed to showcase the skills required for a Junior Machine Learning Engineer role. It follows a clean project structure and includes model training, evaluation, experiment tracking with MLflow, and inference through a FastAPI REST API.

## Featuresx`

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
````markdown

## Data Preparation

The dataset is split into three subsets:

| Split | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Used to train the neural network |
| Validation | 15% | Used to monitor model performance and select the best model |
| Test | 15% | Used for final evaluation on unseen data |

A stratified split is used to preserve the class distribution across
the training, validation, and test sets.

A fixed random seed (`random_state=42`) is used to make the dataset
split reproducible across runs.

### Data Splitting Strategy

```text
Original Dataset (100%)
          |
          v
    +-----+------+
    |            |
    v            v
Training       Temporary
  70%            30%
                  |
            +-----+-----+
            |           |
            v           v
       Validation     Test
          15%          15%
````

The test set is kept separate from the training process and is used
only for final model evaluation.

```
```

---

````markdown
## Model Architecture

The project uses a simple fully connected neural network (MLP)
as a baseline classifier.

```text
Input Image
20 × 20 pixels
     |
     v
Flatten
400 features
     |
     v
Linear Layer
400 → 25
     |
     v
Sigmoid
     |
     v
Linear Layer
25 → 10
     |
     v
10 Output Logits
     |
     v
Digits 0-9
````

### Architecture Details

| Layer         | Configuration    |
| ------------- | ---------------- |
| Input         | 400 features     |
| Hidden Layer  | 25 neurons       |
| Activation    | Sigmoid          |
| Output Layer  | 10 neurons       |
| Output        | Raw logits       |
| Loss Function | CrossEntropyLoss |

The output layer produces 10 logits corresponding to the digit
classes 0 through 9.

Softmax is not explicitly applied in the model because
`CrossEntropyLoss` expects raw logits and handles the required
normalization internally.

The fully connected network is used as a baseline before exploring
more advanced architectures such as convolutional neural networks.

```
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
