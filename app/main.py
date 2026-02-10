import __main__ 
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Engineered features
        X['Loan_to_Income'] = X['LoanAmount'] / X['Income']
        X['Employment_Stability'] = X['MonthsEmployed'] / 12
        X['CreditLines_per_Year'] = X['NumCreditLines'] / (X['Employment_Stability'] + 0.1)
        X['High_Risk_Loan'] = ((X['DTIRatio'] > 0.6) | (X['CreditScore'] < 500)).astype(int)
        return X

__main__.FeatureEngineer = FeatureEngineer

app = FastAPI(title="Loan_payment_prediction" ,
               description="This API predicts whether a loan payment will be made on time or not based on various features.",
                 version="1.0")

# load the pre-trained model to use for predictions
model = joblib.load("best_model_Logistic_Regression.joblib")

# structure of the input data for prediction
class LoanApplication(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str 

@app.post("/predict")    
async def predict(data: LoanApplication):
    # Convert Pydantic model to DataFrame [cite: 2, 4]
    df = pd.DataFrame([data.model_dump()]) 
    
    # Run prediction through the pipeline [cite: 8]
    prediction = model.predict(df)
    probabilities = model.predict_proba(df)

    return {
        "is_high_risk": int(prediction[0]),
        "probability": float(probabilities[0][1]) # Probability of class 1 (High Risk) [cite: 9]
    }
