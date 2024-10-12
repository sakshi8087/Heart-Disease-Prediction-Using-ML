# Heart Disease Prediction Project Documentation

## Table of Contents

1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
   - [Creating a Virtual Environment](#1-create-a-virtual-environment)
   - [Activating the Virtual Environment](#2-activate-the-virtual-environment)
   - [Installing Required Packages](#3-install-required-packages)
   - [Running the Application](#4-running-the-application)
3. [Project Structure](#project-structure)
4. [Categorical Variable Encoding](#categorical-variable-encoding)
5. [Understanding .pkl Files](#understanding-pkl-files)
   - [How it Works](#how-it-works)
6. [Application Features](#application-features)
7. [Conclusion](#conclusion)

## Overview

This project is a Heart Disease Prediction web application built using Flask. It utilizes a Random Forest Classifier model trained on a heart disease dataset to predict whether an individual has heart disease based on various health parameters. The application allows users to input their health data and receive a prediction along with the associated probability.

## Setup Instructions

### 1. Create a Virtual Environment

Before running the application, create a virtual environment to manage your project dependencies. You can do this using the following command:

```bash
--python -m venv myvenv
--myvenv\Scripts\activate

**Installed Packages**
pip install flask numpy pandas scikit-learn joblib

**Project Structure**
heart_disease_prediction/
│
├── app.py                   # Main application file
├── heart.csv                # Dataset used for training
├── label_encoders/          # Directory containing label encoders for categorical features
│   ├── Sex_encoder.pkl      # Label encoder for Sex
│   ├── ChestPainType_encoder.pkl  # Label encoder for ChestPainType
│   ├── RestingECG_encoder.pkl  # Label encoder for RestingECG
│   ├── ExerciseAngina_encoder.pkl  # Label encoder for ExerciseAngina
│   └── ST_Slope_encoder.pkl  # Label encoder for ST_Slope
├── heart_disease_model.pkl  # Trained Random Forest model
├── templates/               # Directory for HTML templates
│   ├── index.html           # Input form
│   └── result.html          # Prediction results page
└── requirements.txt         # Required packages


**Categorical Variable Encoding**
---In this project, we create separate files for each categorical variable's Label Encoder. This approach allows for:
1. Consistent Encoding: Each categorical variable can be encoded independently, ensuring that the model recognizes them accurately.
2. Flexibility: It makes it easier to update or modify the encoding for individual features without affecting others.

**Understanding .pkl Files**
.pkl files are serialized files used to store Python objects, including machine learning models and encoders. The joblib library is used to save and load these objects efficiently.

**How it Works**
Saving Models/Encoders: After training the model or fitting the Label Encoder, we save it as a .pkl file using joblib.dump(). This allows for quick loading later without needing to retrain the model.


**joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')**


**Loading Models/Encoders: When the application starts, we load the saved models and encoders from the .pkl files using joblib.load(). This enables the application to make predictions using the pre-trained model.**

model = joblib.load('heart_disease_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')


**Application Features**
User Input: Users can enter their health parameters, such as age, sex, chest pain type, blood pressure, cholesterol levels, etc.
Prediction: The application predicts whether the user has heart disease and provides the probability of the prediction.
Precautions: If the prediction indicates heart disease, the application suggests precautions and lifestyle changes to help manage heart health.


**Conclusion
This Heart Disease Prediction project showcases how to build a machine learning model with a Flask web application, handling categorical variables effectively while utilizing serialization for model persistence. The project not only demonstrates the technical aspects of machine learning and web development but also serves as a practical tool for predicting heart disease risk based on user-inputted health data.**


**Screenshort**
![image](https://github.com/user-attachments/assets/9ccc97c4-c643-477d-9463-da6f73bc981a)

![image](https://github.com/user-attachments/assets/f75b3c6a-538a-4900-b20c-1117f820547f)




Feel free to customize any sections to better fit your project or add any additional information you think is necessary


