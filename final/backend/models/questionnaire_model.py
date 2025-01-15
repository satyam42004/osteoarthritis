import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Synthetic Data Generation Function
def generate_synthetic_data(num_samples=500):
    """
    Generate synthetic dataset for osteoarthritis prediction based on questionnaire data.
    """
    np.random.seed(42)  # For reproducibility

    # Generate features
    data = {
        "Age": np.random.randint(30, 80, size=num_samples),  # Age range: 30-80
        "Gender": np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5]),  # Male=0, Female=1
        "JointPain": np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7]),  # Pain: 70% likely
        "MorningStiffness": np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]),  # 60% likely
        "Swelling": np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5]),  # Equal likelihood
        "RangeOfMotion": np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]),  # 60% likely
        "SedentaryLifestyle": np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5]),  # Equal likelihood
        "ActivityInducedPain": np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7]),  # 70% likely
        "InjuryHistory": np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8]),  # 20% likely
        "FamilyHistory": np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]),  # 60% likely
        "Overweight": np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5]),  # Equal likelihood
        "Diet": np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4]),  # Poor diet: 60% likely
    }

    # Generate target variable (HasOA) based on a scoring function
    def generate_target(row):
        # Combine risk factors for likelihood of OA
        risk_score = (
            (row["Age"] > 50) * 2
            + row["JointPain"]
            + row["MorningStiffness"]
            + row["Swelling"]
            + row["RangeOfMotion"]
            + row["SedentaryLifestyle"]
            + row["ActivityInducedPain"]
            + row["InjuryHistory"]
            + row["FamilyHistory"]
            + row["Overweight"]
            + (1 - row["Diet"])  # Poor diet increases risk
        )
        return 1 if risk_score > 6 else 0

    # Create DataFrame
    df = pd.DataFrame(data)
    df["HasOA"] = df.apply(generate_target, axis=1)

    # Ensure the directory exists
    output_dir = "backend/models/data"
    os.makedirs(output_dir, exist_ok=True)

    # Save the dataset
    output_path = os.path.join(output_dir, "synthetic_osteoarthritis_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset generated and saved to '{output_path}'.")
    return df

# Predict Function
def predict_oa(input_data):
    """
    Predict whether a person has OA based on questionnaire data.
    :param input_data: DataFrame containing user input
    :return: Prediction (1 = High Probability of OA, 0 = Low Probability of OA)
    """
    # Risk scoring (same logic as target generation)
    def calculate_risk(row):
        risk_score = (
            (row["Age"] > 50) * 2
            + row["JointPain"]
            + row["MorningStiffness"]
            + row["Swelling"]
            + row["RangeOfMotion"]
            + row["SedentaryLifestyle"]
            + row["ActivityInducedPain"]
            + row["InjuryHistory"]
            + row["FamilyHistory"]
            + row["Overweight"]
            + (1 - row["Diet"])
        )
        return 1 if risk_score > 6 else 0

    # Apply risk calculation
    prediction = input_data.apply(calculate_risk, axis=1)
    return prediction

# Main Execution
if __name__ == "__main__":
    # Generate Synthetic Data
    df = generate_synthetic_data()

    # Split data into training and testing sets
    X = df.drop("HasOA", axis=1)
    y = df["HasOA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict on the test set
    y_pred = predict_oa(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Example Prediction
    print("\n--- Example Prediction ---")
    example_input = {
        "Age": 60,
        "Gender": 1,
        "JointPain": 1,
        "MorningStiffness": 1,
        "Swelling": 0,
        "RangeOfMotion": 1,
        "SedentaryLifestyle": 1,
        "ActivityInducedPain": 1,
        "InjuryHistory": 0,
        "FamilyHistory": 1,
        "Overweight": 0,
        "Diet": 0,
    }
    example_df = pd.DataFrame([example_input])
    prediction = predict_oa(example_df)
    print(f"Prediction (1=High Probability of OA, 0=Low Probability of OA): {prediction.values[0]}")
