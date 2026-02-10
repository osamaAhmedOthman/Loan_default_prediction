# Loan Default Prediction Project

An end-to-end machine learning project to predict loan default risk. The project covers data preprocessing, feature engineering, model training, evaluation, and saving a production-ready inference pipeline.

---

## 📁 Project Structure
```
Loan_default_prediction/
│
├── data/
│   ├── data.csv                         # Original dataset
│   └── data_feature_engineered.csv      # After feature engineering
│
├── app/
│   ├── best_model_Logistic_Regression   # model
│   ├── main.py                          # api file
│   └── requirements.txt      # requirements of the api
│
├── visuals/                             # Plots & visualizations
│
├── notebooks/                         
│   └── Loan_default.ipynb               # Main analysis & modeling notebook
│
├── models/                              
│   └── best_model_Logistic_Regression.joblib   # Final saved ML pipeline
│
├── src/                                 # Source code
│   ├── feature_engineer.py              # Custom feature engineering class
│   ├── pipeline_test.py                     # Script to load & test pipeline
│ 
├── Reports/                              # Project reports
│   └── Loan_Default_Prediction_Report.pdf   # Full project report
│
├── __pycache__/                         # Auto-generated Python cache
│
├── README.md                            # Project documentation
└── requirements.txt                      # Package dependencies
└── .gitignore                            # Git ignore file                      
```

---


## 🚀 Workflow Overview
### **1. Data Processing & Feature Engineering**
- Handling missing values
- Scaling numerical features
- Encoding categorical features
- Creating advanced engineered features:
- `Loan_to_Income`
- `Employment_Stability`
- `CreditLines_per_Year`
- `High_Risk_Loan`


All transformations are built into the **FeatureEngineer** class and integrated inside the saved pipeline.


---


## 🤖 Modeling
Trained classification models:
- Logistic Regression
- Random Forest
- XGBoost
- Naive Bayes
- Decision Tree


Evaluated using:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Sensitivity & Specificity


🏆 **Logistic Regression** delivered the best ROC-AUC and was saved as the final inference pipeline.


---


## 📦 Pipeline Deployment
The file `best_model_Logistic_Regression.joblib` contains a full pipeline:
- Feature engineering
- Preprocessing (imputation, scaling, encoding)
- Final trained model


➡️ **You can run predictions directly on raw input data. No manual preprocessing required.**


---


## ▶️ Running Predictions
Execute:
```
python pipeline_test.py
```
This script:
- Loads the saved pipeline
- Passes a test sample
- Returns prediction + probability


---


## 📌 Author
**Osama Othman**
📩 Email: **osmanosamaahmed@gmail.com**
🔗 LinkedIn: **https://www.linkedin.com/in/osama-othman-a78141368/**
