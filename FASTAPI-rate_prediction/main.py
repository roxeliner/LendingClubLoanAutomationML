from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import lightgbm as lgb

# Load your trained model and preprocessor
model = joblib.load("tuned_lgbm_rate_model.pkl")
preprocessor = joblib.load("preprocessor_rate.pkl")

app = FastAPI()


# Define a BaseModel for the input data
class LoanData(BaseModel):
    loan_amnt: float
    term: str
    loan_status: str
    sub_grade: str
    bc_util: float
    fico_range_high: int
    revol_util: float
    emp_length: str
    home_ownership: str
    addr_state: str
    fico_range_low: int
    inq_last_6mths: int
    annual_inc: float
    bc_open_to_buy: float
    dti: float
    mths_since_recent_inq: int
    total_acc: int
    num_op_rev_tl: int
    grade: str

# A function to classify loan status
def classify_loan_status(status):
    if status == "Fully Paid":
        return 1
    elif status == "Charged Off":
        return 0
    else:
        return np.nan

# Include your feature creation function
def create_features(df):
    df["application_status"] = df["loan_status"].apply(classify_loan_status)
    df.dropna(axis=0, inplace=True)
    df["term_months"] = df["term"].str.extract("(\d+)").astype(int)
    df["emp_length_years"] = df["emp_length"].replace({"10+ years": "10", "< 1 year": "0"})
    df["emp_length_years"] = df["emp_length_years"].str.extract("(\d+)").astype(float)
    df["utilization_rate"] = df["bc_util"] / 100
    df["fico_score_avg"] = (df["fico_range_high"] + df["fico_range_low"]) / 2
    df["dti_category"] = pd.cut(df["dti"], bins=[0, 10, 20, 30, 40, 50, 100], labels=False)
    df["log_annual_inc"] = np.log(df["annual_inc"] + 1)
    df["loan_income_ratio"] = df["loan_amnt"] / df["annual_inc"]
    df["short_term"] = (df["term_months"] <= 36).astype(int)
    df["years_since_last_inquiry"] = df["mths_since_recent_inq"] / 12
    df["credit_line_util"] = df["num_op_rev_tl"] / df["total_acc"]
    df["fico_score_category"] = pd.cut(df["fico_score_avg"], bins=[0, 600, 650, 700, 750, 800, 850], labels=False)
    df["home_ownership_own"] = (df["home_ownership"] == "OWN").astype(int)

    return df


# Endpoint to predict loan status
@app.post("/interest_rate/")
async def predicted_interest_rate(data: LoanData):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Add a int_rate column with a default value, needed for create_features
        input_data['int_rate'] = 12

        # Apply the same feature engineering as in the training
        input_data = create_features(input_data)

        # Preprocess the data
        input_data_prepared = preprocessor.transform(input_data)

        # Predict and return the prediction
        int_rate  = model.predict(input_data_prepared)
        return {"predicted_interest_rate": int_rate[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
