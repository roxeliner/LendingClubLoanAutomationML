import requests

# The endpoint URL of your FastAPI application
endpoint_url = 'https://loanapp-lbydg4vitq-uc.a.run.app/predict/'

# Replace this with the actual data extracted from the image
data_to_predict = {
    "loan_amnt": 3600.0,
    "term": "36 months",
    "int_rate": 13.99,
    "sub_grade": "C4",
    "bc_util": 37.2,
    "fico_range_high": 679.0,
    "revol_util": 29.7,
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "addr_state": "PA",
    "fico_range_low": 675.0,
    "inq_last_6mths": 1.0,
    "annual_inc": 55000.0,
    "bc_open_to_buy": 1506.0,
    "dti": 5.91,
    "mths_since_recent_inq": 4.0,
    "total_acc": 13.0,
    "num_op_rev_tl": 4.0
}

# Send a POST request with the JSON payload
response = requests.post(endpoint_url, json=data_to_predict)

# Check the response
if response.status_code == 200:
    print("Prediction successful:", response.json())
else:
    print("Prediction failed with status code:", response.status_code)
