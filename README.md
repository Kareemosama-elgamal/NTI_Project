# NTI_Project
📊 Adult Income Classification – Machine Learning Project

A machine learning project that predicts whether a person's income exceeds $50K/year based on demographic and employment data, using the Adult Income Dataset.

🔍 Overview

This project covers a full data science workflow — from data cleaning and preprocessing to training and evaluating multiple classification models.

⚙️ Steps & Techniques
Data Cleaning: Removing duplicates, handling missing values and unknown entries (?)
Categorical Encoding: Converting text columns using LabelEncoder
Correlation Analysis: Using heatmaps to explore relationships between features and the income target
Feature Scaling: Applying MinMaxScaler to numerical columns
Outlier Handling: Using the IQR (Interquartile Range) me…
[01:28, 28/07/2026] +20 12 03829208: Readme · MD
# 📊 Adult Income Classification – Machine Learning Project
 
A machine learning project that predicts whether a person's income exceeds $50K/year based on demographic and employment data, using the Adult Income Dataset.
 
## 🔍 Overview
 
This project covers a full data science workflow — from data cleaning and preprocessing to training and evaluating multiple classification models.
 
## ⚙️ Steps & Techniques
 
- *Data Cleaning:* Removing duplicates, handling missing values and unknown entries (?)
- *Categorical Encoding:* Converting text columns using LabelEncoder
- *Correlation Analysis:* Using heatmaps to explore relationships between features and the income target
- *Feature Scaling:* Applying MinMaxScaler to numerical columns
- *Outlier Handling:* Using the IQR (Interquartile Range) method for capping/removal
- *Model Training:*
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN), with automatic selection of the best K value
  - Decision Tree Classifier (tuned with GridSearchCV)
  - Logistic Regression
- *Model Evaluation:* Accuracy, Precision, Recall, F1-score, Confusion Matrix, and Classification Report for each model
## 🧰 Tech Stack
 
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
 
## 📁 Dataset
 
Adult Census Income dataset (adult.csv) — predicts income category based on features like age, education, occupation, work hours, and more.
