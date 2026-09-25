# Titanic Survival Classification

## Project Overview

This project uses the Titanic dataset from Kaggle to build a Machine Learning classification model that predicts whether a passenger survived or did not survive.

The project focuses on data cleaning, preprocessing, feature selection, model training, and model evaluation.

## Objective

The main objective is to predict passenger survival based on features such as:

- Passenger class
- Gender
- Age
- Fare
- Embarkation port

## Dataset

The dataset is taken from the Kaggle Titanic: Machine Learning from Disaster competition.

Dataset:
https://www.kaggle.com/competitions/titanic

The project uses the `train.csv` file because it contains the `Survived` target column required for model training.

## Features Used

| Feature | Description |
|---|---|
| Pclass | Passenger ticket class |
| Sex | Passenger gender |
| Age | Passenger age |
| Fare | Ticket fare |
| Embarked | Port where the passenger boarded |

## Target

The target variable is:

`Survived`

- `0` = Did not survive
- `1` = Survived

## Machine Learning Model

The project uses:

**Logistic Regression**

Logistic Regression is a classification algorithm used to predict the probability of a class.

## Project Workflow

The project follows these steps:

1. Load the dataset
2. Explore the data
3. Check missing values
4. Select relevant features
5. Handle missing values
6. Encode categorical features
7. Split the data into training and testing sets
8. Train the Logistic Regression model
9. Make predictions
10. Evaluate model performance

## Data Preprocessing

The following preprocessing techniques are used:

- Missing Age values are replaced using the median.
- Missing Embarked values are replaced using the mode.
- Gender values are converted into numerical values.
- Embarked values are converted using one-hot encoding.

## Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

A `random_state` of 42 is used to make the split reproducible.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Project Structure

```text
task_1_titanic_survival_classification/
│
├── data/
│   └── raw/
│       └── train.csv
│
├── notebooks/
│   └── titanic_analysis.ipynb
│
├── src/
│   └── titanic_classification.py
│
├── results/
│
├── README.md
└── requirements.txt