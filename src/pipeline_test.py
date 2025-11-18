# pipeline_test.py

import pandas as pd
import joblib
import argparse
from src.feature_engineer import FeatureEngineer   


# 1) Load the saved pipeline
pipeline_file = "best_model_Logistic_Regression.joblib"
pipeline = joblib.load(pipeline_file)
print("✅ Loaded pipeline successfully!")



# 2) Function to predict from a dictionary (single example)
def predict_single():
    data = pd.DataFrame([
        {
            'Age': 35,
            'Income': 50000,
            'LoanAmount': 15000,
            'CreditScore': 650,
            'MonthsEmployed': 60,
            'NumCreditLines': 5,
            'InterestRate': 0.08,
            'LoanTerm': 36,
            'DTIRatio': 0.3,
            'Education': 'Bachelors',
            'EmploymentType': 'Salaried',
            'MaritalStatus': 'Married',
            'HasMortgage': 'No',
            'HasDependents': 'Yes',
            'LoanPurpose': 'Home',
            'HasCoSigner': 'No'
        }
    ])

    pred = pipeline.predict(data)
    prob = pipeline.predict_proba(data)[:, 1]

    print("\n📌 Prediction for Single Input:")
    print("Class (0=No Default, 1=Default):", int(pred[0]))
    print("Default Probability:", float(prob[0]))


# 3) Function to predict from CSV file
def predict_from_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"\n📄 Loaded CSV with {df.shape[0]} rows.")

    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)[:, 1]

    df["Prediction"] = preds
    df["Probability"] = probs

    output_file = "predictions_output.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ Predictions saved to: {output_file}")


# 4) Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Default Prediction Tester")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file containing new loan data"
    )

    args = parser.parse_args()

    if args.csv:
        predict_from_csv(args.csv)  # CSV mode
    else:
        predict_single()            # single prediction mode
