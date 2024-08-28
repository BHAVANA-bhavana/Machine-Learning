# install libraries ---
# pip install fastapi uvicorn 

# 1. Library imports
# import uvicorn
# from fastapi import FastAPI

# from fastapi.middleware.cors import CORSMiddleware
# import pickle

# 2. Create the app object
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# 3. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index():
#     return {'message': 'Hello, World'}

# 5. Run the API with uvicorn
# if __name__ == '__main__':
#     uvicorn.run(app, port=80, host='0.0.0.0')

##---------------------
# 3. load the model
# log = pickle.load(open("MobilePrice_Dataset_log.pkl", "rb"))

# @app.get("/predictPrice")
# def gePredictPrice(ram: int, battery_power: int, px_width: int, px_height: int):
#     prediction = log.predict([[ram,battery_power,px_width,px_height]])
#     return {'Price': prediction[0]}
    
# uvicorn app:app --reload
# uvicorn app:app --host 0.0.0.0 --port 80
# http://127.0.0.1/predictPrice?Area=1400&BedRooms=3&BathRooms=3

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = pickle.load(open("out_put.pkl", "rb"))

# Map numerical classes to their labels
class_labels = {
    0: "High-Value Shoppers",
    1: "Young Affluent Spenders",
    2: "Middle-Income Shoppers",
    3: "Budget-Conscious Seniors"
}

# Define the endpoint for predicting mobile prices
@app.get("/predictCluster")
def gePredictPrice(Age:int=19,Annual_Income:int=23,Spending_Score:int=1):
    prediction = model.predict([[Age,Annual_Income,Spending_Score]])
    predicted_class = int(prediction[0])  # Ensure the prediction is an integer
    cluster_label = class_labels.get(predicted_class, 'Unknown')
    return {'Cluster': cluster_label}

# Index route
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Define the endpoint for fetching cluster labels
@app.get("/clusterLabels")
def getClusterLabels():
    return class_labels

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
