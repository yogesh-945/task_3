# task- 3 Heart disease Prediction
# 🫀 Heart Disease Prediction Web App

A simple yet powerful machine learning web application that predicts the likelihood of heart disease based on input health parameters.

Built as part of **Task 3** for the **TechNest Data Science Virtual Internship**.

---

## 📌 Project Overview

This project uses a logistic regression model trained on a heart disease dataset from Kaggle. The model is deployed using **Flask**, allowing users to input clinical data through a simple web interface and receive instant predictions.

---

## 🚀 Demo


🖥️ To run locally:  
Visit → [http://127.0.0.1:5000/](http://127.0.0.1:5000/) after starting the server

---

## 🛠️ Features

- User-friendly web interface
- Predicts heart disease based on health indicators
- Logistic Regression model with good accuracy
- Real-time output on form submission
- Clean UI with Flask backend

---

## 📁 Folder Structure
heart-disease-prediction/
├── app.py # Flask web app
├── model/ 
        model_training.py
        heart.csv (dataset)
├── templates/
│ └── index.html # Frontend form
├── static/
│ └── style.css # Optional styling
|-- utils/ preprocess.py
└── requirements.txt # Dependencies

---

## 💻 How to Run Locally

### ✅ Step 1: Clone the Repository

git clone https://github.com/YOUR-USERNAME/heart-disease-prediction.git
cd heart-disease-prediction

Step 2: Create a Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate      # For Windows
# OR
source venv/bin/activate   # For Mac/Linux

✅ Step 3: Install Dependencies
pip install -r requirements.txt

✅ Step 4: Run the App
python app.py

Now, open your browser and go to:
👉 http://127.0.0.1:5000/

✨ Sample Inputs
Feature	Example
Age	52
Sex	Male
Cholesterol	220
Resting BP	130
FastingBS	1
Max HR	160
Chest Pain	ATA



