import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from parameters import params
app = FastAPI()
pickle_in = open('regmodel.pkl','rb')
regmodel = pickle.load(pickle_in)
@app.get('/')
def index():
    return{'message':'Hello'}
@app.get('/{name}')
def get_name(name:str):
    return {'message':f'Hello,{name}'}
@app.post('/predict')
def predict_func(data:params):
    data = data.dict
    texture_mean=data['texture_mean']
    smoothness_mean=data['smoothness_mean']
    compactness_mean=data['compactness_mean']
    concave_points_mean=data['concave_points_mean']
    symmetry_mean=data['symmetry_mean']
    fractal_dimension_mean=data['fractal_dimension_mean']
    texture_se=data['texture_se']
    area_se=data['area_se']
    smoothness_se=data['smoothness_se']
    compactness_se=data['compactness_se']
    concavity_se=data['concavity_se']
    concave_points_se=data['concave_points_se']
    symmetry_se=data['symmetry_se']
    fractal_dimension_se=data['fractal_dimension_se']
    texture_worst=data['texture_worst']
    area_worst=data['area_worst']
    smoothness_worst=data['smoothness_worst']
    compactness_worst=data['compactness_worst']
    concavity_worst=data['concavity_worst']
    concave_points_worst=data['concave_points_worst']
    symmetry_worst=data['symmetry_wors']
    fractal_dimension_worst=data['fractal_dimension_worst']
    

    prediction = log_reg.predict([[texture_mean, smoothness_mean, compactness_mean,
       concave_points_mean, symmetry_mean, fractal_dimension_mean,
       texture_se, area_se, smoothness_se, compactness_se,
       concavity_se, concave_points_se, symmetry_se,
       fractal_dimension_se, texture_worst, area_worst,
       smoothness_worst, compactness_worst, concavity_worst,
       concave_points_worst, symmetry_worst, fractal_dimension_worst]])
    
    if(prediction==1):
        prediction="cancerous"
    else:
        prediction="non-cancerous"
    return {
        'prediction': prediction
    }        

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
