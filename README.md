# Legal Document Classification

## Introduction
This repository contains a machine learning project for classifying legal documents into different categories. The goal of this project is to automatically classify legal documents based on their text content. Various machine learning algorithms and techniques are employed to achieve accurate classification results.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Introduction
In this project, we aim to classify legal documents into different categories based on their text content. The goal is to develop a machine learning model that can automatically assign the correct category to a given legal document. We explore various machine learning algorithms and techniques to achieve accurate classification results.

## Dataset
The dataset used in this project consists of legal documents from different categories such as contracts, patents, court cases, and legal opinions. Each document is labeled with its corresponding category. The dataset is preprocessed and prepared for training and evaluation.

## Usage
To use this code, follow these steps:

1. Ensure that the dataset file is available in the same directory as the code file.
2. Import the required libraries:
   - `from sklearn.feature_extraction.text import TfidfVectorizer`: To convert text documents into numerical features using TF-IDF representation.
   - `from sklearn.model_selection import train_test_split, cross_val_score`: To split the dataset into training and testing sets and perform cross-validation.
   - `from sklearn.naive_bayes import MultinomialNB`: To implement the Naive Bayes classifier.
   - `from sklearn.svm import SVC`: To implement the Support Vector Machine classifier.
   - `from sklearn.tree import DecisionTreeClassifier`: To implement the Decision Tree classifier.
   - `from sklearn.ensemble import RandomForestClassifier`: To implement the Random Forest classifier.
   - `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer`: To evaluate the model's performance using various metrics.
   - `import pandas as pd`: For data manipulation and analysis.
   - `import numpy as np`: For numerical operations.
   - `import re`: For text preprocessing.
   - `import nltk`: For natural language processing tasks.
   - `from nltk.corpus import stopwords`: To remove stopwords from text data.
   - `from sklearn.model_selection import KFold`: For K-fold cross-validation.
   - `from sklearn.model_selection import cross_validate`: To perform cross-validation.
   - `from sklearn.linear_model import LogisticRegression`: To implement the Logistic Regression classifier.
   - `from gensim.models import Word2Vec`: For word embedding techniques.

3. Load the dataset and perform any necessary preprocessing steps.
4. Split the dataset into training and testing sets using `train_test_split()`.
5. Create a `TfidfVectorizer` object to convert text documents into numerical features using TF-IDF representation.
6. Choose a machine learning algorithm (e.g., Naive Bayes, Support Vector Machine, Decision Tree, Random Forest, Logistic Regression).
7. Train the selected model on the training data.
8. Evaluate the model's performance using various metrics such as accuracy, precision, recall, and F1-score.
9. Make predictions on new unseen legal documents using the trained model.

Feel free to experiment with different algorithms, feature extraction techniques, and parameter settings to find the best approach for legal document classification.


## Acknowledgments
- The legal document dataset used in this repository is aavailable on KAGGLE
