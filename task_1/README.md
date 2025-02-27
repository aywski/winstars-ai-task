# Image Classification and OOP

## Overview
This project implements a simple image classification system for the MNIST dataset using three different models:
1. **Random Forest Classifier (RF)** 
2. **Feed-Forward Neural Network (NN)** 
3. **Convolutional Neural Network (CNN)** 

## Steps to Run the notebook:
1. Clone the repository.
```bash
git clone https://github.com/aywski/winstars-ai-task
```
2. Create a virtual environment: 
```bash
cd task2
python -m venv .venv
.venv\Scripts\activate    # on Windows 
source .venv/bin/activate   # on Linux
```
3. Install dependencies: `pip install -r requirements.txt`.
4. Use the `notebook.ipynb` to validate this task:

## Repository Structure
```
task_1/
│
├── models/
│   ├── cnn_model.py
│   ├── nn_model.py
│   └── rf_model.py
│
├── interface.py
├── mnist_classifier.ipynb
├── notebook.ipynb
├── README.md
└── requirements.txt
```

## Scripts Overview
#### 1. `models/cnn_model.py`
- Implementation of the CNN model
#### 2. `models/nn_model.py`
- Implementation of the NN model
#### 3. `models/rf_model.py`
- Implementation of the RF model
#### 4. `interface.py`
- Implementation of abstract methods train and predict.
#### 5. `mnist_classifier.py`
- This file contains a method designed to allow flexible classification using different algorithms.
#### 6. `notebook.ipynb`
- The main notebook used to carry out all machine learning calculations.

## Requirements
- __TensorFlow__
- numpy, scikit-learn, matplotlib
- Python 3.12.4
