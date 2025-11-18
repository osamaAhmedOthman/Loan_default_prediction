import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Recompute engineered features exactly as during training
        X['Loan_to_Income'] = X['LoanAmount'] / X['Income']
        X['Employment_Stability'] = X['MonthsEmployed'] / 12
        X['CreditLines_per_Year'] = X['NumCreditLines'] / (X['Employment_Stability'] + 0.1)
        X['High_Risk_Loan'] = ((X['DTIRatio'] > 0.6) | (X['CreditScore'] < 500)).astype(int)

        return X
