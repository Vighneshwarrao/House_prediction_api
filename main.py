from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import pickle
import numpy as np
with open('Model/model.pkl' , 'rb') as f:
    mp=pickle.load(f)

with open('Model/columns.json','r') as f:
    cols=json.load(f)['data_columns']

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def hello():
    return {"message":"Hello"}

@app.get('/predict')
def predict_price(location : str, sqft : float, bath : float, bhk : float, area_type : int):

    x = np.zeros((len(cols)))
    x[cols.index('total_sqft')] = sqft
    x[cols.index('bath')] = bath
    x[cols.index('bhk')] = bhk
    x[cols.index('area_type_encoded')] = area_type
    if location in cols:
        loc_index=cols.index(location)
        x[loc_index] = 1
    price=mp.predict([x])[0]
    return{'price':round(price,2)}
