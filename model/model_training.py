mport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import pickle
import os

def train_and_save_model():
    df = pd.read_csv(r'C:\Users\Yogesh Verma\Desktop\Educational Content\intern project\heart_disease_project\model\heart.csv')  # <- Use relative path

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numerical_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder='passthrough')  # Pass numerical features through

    # Create final pipeline with model
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier())
    ])

    # Train pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Model Evaluation:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs("model", exist_ok=True)
    with open("model/heart_disease_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
