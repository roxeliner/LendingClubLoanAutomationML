import requests

# The endpoint URL of your FastAPI application
endpoint_url = 'https://interestrateapp-lbydg4vitq-nw.a.run.app'

# Replace this with the actual data extracted from the image
data_to_predict = {
    "loan_amnt": 10000.0,
    "term": "36 months",
    "loan_status": "Fully Paid",
    "sub_grade": "C1",
    "bc_util": 50.0,
    "fico_range_high": 700,
    "revol_util": 30.0,
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "addr_state": "CA",
    "fico_range_low": 690,
    "inq_last_6mths": 1,
    "annual_inc": 80000.0,
    "bc_open_to_buy": 12000.0,
    "dti": 15.0,
    "mths_since_recent_inq": 5,
    "total_acc": 25,
    "num_op_rev_tl": 10,
    "grade": "B"
}


# Send a POST request with the JSON payload
response = requests.post(f"{endpoint_url}/interest_rate/",
                         json=data_to_predict)

# Check the response
if response.status_code == 200:
    print("Prediction successful:", response.json())
else:
    print("Prediction failed with status code:", response.status_code)
