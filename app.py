import uvicorn
from fastapi import FastAPI
from diabetes import diabetes
import  numpy as np 
import pickle
import pandas as pd 



app = FastAPI()
pickle_in = open("svc.pkl","rb")
svc=pickle.load(pickle_in)


@app.post("/predict")
def predict_banknote(data:diabetes):
    data = data.dict()
    Pregnancies = data["Pregnancies"]
    Glucose = data["Glucose"]
    BloodPressure = data["BloodPressure"]
    SkinThickness = data["SkinThickness"]
    Insulin = data["Insulin"]
    BMI = data["BMI"]
    DiabetesPedigreeFunction = data["DiabetesPedigreeFunction"]
    Age = data["Age"]
    
    prediction = svc.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])  
    
    if(prediction[0]==0):
        prediction = "No tiene diabetes"
    else:
        prediction = "Tiene diabetes" 
        
    return{"prediction":prediction} 

