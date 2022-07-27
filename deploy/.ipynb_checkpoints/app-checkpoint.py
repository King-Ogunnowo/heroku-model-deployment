import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl
from dependency import constants as con
from dependency import functions as func

model = func.load_pickle_obj("./model/model_pipeline.pkl")

st.title("Heart Disease Predictor")
st.markdown("A web app to help you predict the likelihood of a **heart attack**")
st.text("please input the following details")
st.sidebar.markdown('[![Oluwaseyi-Ogunnowo]'
                        '(https://img.shields.io/badge/Contact%20Details-Oluwaseyi%20Ogunnowo-blue)]'
                        '(https://www.linkedin.com/in/oluwaseyi-ogunnowo/)')



BMI = st.number_input("What is your Body Mass Index (BMI)?", min_value = 0.0, max_value = 60.0, value=20.5, format="%.2f")
Smoking = st.selectbox("Do you smoke?", ('Yes', 'No'))
AlcoholDrinking = st.selectbox("Do you consume alcohol drinks?", ('Yes', 'No'))
Stroke = st.selectbox("Have you ever had stroke", ('Yes', 'No'))
PhysicalHealth = st.number_input("In a month, how frequently do you worry about your physical health (input number of days)", min_value = 0, max_value = 30, value = 0)
MentalHealth = st.number_input("In a month, how frequently do you worry about your mental health (input number of days)", min_value = 0, max_value = 30, value = 0)
DiffWalking = st.selectbox("Do you in anyway experience difficulty walking?", ('Yes', 'No'))
Sex = st.selectbox("What gender do you identify as", ("Male", "Female"))
AgeCategory = st.selectbox("Please indicate your age category", ('18-24','25-29','30-34','35-39',
                                                                   '40-44','45-49','50-54','55-59',
                                                                   '60-64','65-69','70-74','75-79',
                                                                   '80 or older'))
Race = st.selectbox("What race do you identify as?", ('American Indian/Alaskan Native','Asian','Black',
                                                        'Hispanic','Other','White'))
Diabetic = st.selectbox("Are you diabetic?", ("Yes", "No"))
PhysicalActivity = st.selectbox("Do you engage in physical or athletic activities?", ("Yes", "No"))
GenHealth = st.selectbox("What's your general health perspective?", ('Poor', 'Fair', 'Good', 'Very good','Excellent'))
SleepTime = st.number_input("What's your bedtime? (enter hour value only)", min_value = 0, max_value = 24, value = 20)
Asthma = st.selectbox("Do you have asthmna?", ("Yes", "No"))
KidneyDisease = st.selectbox("Do you currently have any form of Kidney Disease?", ("Yes", "No"))
SkinCancer = st.selectbox("Do you currently have any form of skin cancer?", ("Yes", "No"))

data = pd.DataFrame({'BMI':BMI, 
                     'Smoking':Smoking,
                     'AlcoholDrinking':AlcoholDrinking, 
                     'Stroke':Stroke, 
                     'PhysicalHealth':PhysicalHealth, 
                     'MentalHealth':MentalHealth, 
                     'DiffWalking':DiffWalking, 
                     'Sex':Sex, 
                     'AgeCategory':AgeCategory, 
                     'Race':Race, 
                     'Diabetic':Diabetic, 
                     'PhysicalActivity':PhysicalActivity, 
                     'GenHealth':GenHealth, 
                     'SleepTime':SleepTime, 
                     'Asthma':Asthma, 
                     'KidneyDisease':KidneyDisease, 
                     'SkinCancer':SkinCancer}, index = [0])    

    
if st.button("👌 Predict Likelihood"):
    probability = func.predict(data, model)
    st.success(f'You have a {int(probability)}% chance of developing heart disease')
    if probability > 50:
        st.warning("Please go for medical checkup")
    





