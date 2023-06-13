from pydantic import BaseModel

class diabetes(BaseModel):
    Pregnancies: int 
    Glucose: int 
    BloodPressure: int 
    SkinThickness: int 
    Insulin: int 
    BMI: float
    DiabetesPedigreeFunction: float 
    Age: int 