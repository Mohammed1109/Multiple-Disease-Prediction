# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:29:29 2024

@author: HP
"""
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


#with open(r"C:\Users\HP\Multiple Disease\style.css") as f:
#  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)




Diabetes_model = pickle.load(open(r"C:\Users\HP\Multiple Disease\Diabetes Disease\trained_model.sav", 'rb'))
Parkinsons_model = pickle.load(open(r"C:\Users\HP\Multiple Disease\Parkinson Disease\Park_trained_model.sav", 'rb'))
Heart_model = pickle.load(open( r"C:\Users\HP\Multiple Disease\Heart Disease\Heart_trained_model.sav", 'rb'))


Diabetes_scalar = pickle.load(open("C:/Users/HP/Multiple Disease/Diabetes Disease/scalar.sav", 'rb'))
# Diabetes Prediction function
def diabetes_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data).astype(float)
    
    newdata = input_data_as_numpy_array.reshape(1, -1)
    
  # Use the loaded scalar to transform the input data
    stand_data = Diabetes_scalar.transform(newdata)

# Use the loaded model to make predictions
    prediction = Diabetes_model.predict(stand_data)
    
    feature_importances = Diabetes_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'Importance': feature_importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_features = feature_importance_df.head(3)
    top_feature_values = input_data_as_numpy_array[top_features.index]
    
    if prediction == 0:
        diagnosis = "Person is non-diabetic and top 3 features contributing to non-diabetic prediction is "
        for i in range(len(top_features)):
            feature = top_features.iloc[i]
            value = top_feature_values[i]
            diagnosis += f"{feature}: {value}\n"
    else:
        diagnosis = "Person is diabetic and top 3 features contributing to diabetic prediction is "
        for i in range(len(top_features)):
            feature = top_features.iloc[i]
            value = top_feature_values[i]
            diagnosis += f"{feature}: {value}\n"
    
    return diagnosis



Heart_scalar = pickle.load(open("C:/Users/HP/Multiple Disease/Heart Disease/Heart_scalar.sav", 'rb'))
# Heart Disease Prediction function
def heart_prediction(input_data):
  
    input_data_as_numpy_array = np.asarray(input_data).astype(float)
  
    newdata = input_data_as_numpy_array.reshape(1, -1)
    
    stand_data = Heart_scalar.transform(newdata)
    prediction = Heart_model.predict(stand_data)
    
    coefficients = Heart_model.coef_[0]
    feature_coefficients = pd.DataFrame({
        'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'Coefficient': coefficients
    })
    top_features = feature_coefficients['Feature'].head(3)
    top_feature_values = input_data_as_numpy_array[top_features.index]
    
    if prediction == 0:
        diagnosis = "This Person's Heart is Good. Top 3 features contributing to Heart prediction:\n"
        for i in range(len(top_features)):
            feature = top_features.iloc[i]
            value = top_feature_values[i]
            diagnosis += f"{feature}: {value}\n"
    else:
        diagnosis = "Person is having Heart Disease.\nTop 3 features contributing to Heart prediction:\n"
        for i in range(len(top_features)):
            feature = top_features.iloc[i]
            value = top_feature_values[i]
            diagnosis += f"{feature}: {value}\n"
    
    return diagnosis
Parkinson_scalar = pickle.load(open("C:/Users/HP/Multiple Disease/Parkinson Disease/Park_scalar.sav", 'rb'))
# Parkinson's Disease Prediction function
def park_prediction(input_data):
   
    input_data_as_numpy_array = np.asarray(input_data).astype(float)
   
    newdata = input_data_as_numpy_array.reshape(1, -1)
    
    stand_data = Parkinson_scalar.transform(newdata)
    prediction = Parkinsons_model.predict(stand_data)
    
    coefficients = Parkinsons_model.coef_[0]
    feature_coefficients = pd.DataFrame({
        'Feature': ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'],
        'Coefficient': coefficients
    })
    top_features = feature_coefficients['Feature'].head(3)
    top_feature_values = input_data_as_numpy_array[top_features.index]
    
    if prediction == 0:
        diagnosis = "Person is normal. Top 3 features contributing to normal/non-parkinson prediction:\n"
        for i in range(len(top_features)):
            feature = top_features.iloc[i]
            value = top_feature_values[i]
            diagnosis += f"{feature}: {value}\n"
    else:
        diagnosis = "Person is having Parkinson Problem. Top 3 features contributing to Parkinson prediction:\n"
        for i in range(len(top_features)):
            feature = top_features.iloc[i]
            value = top_feature_values[i]
            diagnosis += f"{feature}: {value}\n"
    
    return diagnosis

# Streamlit app main function
def main():
    with st.sidebar:
        selected = option_menu("Multiple Disease Prediction System using Machine Learning", 
                               ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Disease Prediction"],
                               icons=["activity", "heart-fill", "people-fill"],
                               default_index=0 ,
                               styles={
            
            "container": {"padding": "0!important", "background-color": "#cccccc"},  # Gray background
            "nav-link": {"color": "#000000"}, })

    if selected == "Diabetes Prediction":
            st.title('Diabetes Prediction Web App')
            diagnosis = ''
            
            with st.form("diabetes_form"):
                col1, col2 = st.columns(2)
    
                with col1:
                    Pregnancies = st.number_input("Number of Pregnancies(1-17)", min_value=0, max_value=17, step=1, format='%d' , args='widgetargs')
                    BloodPressure = st.number_input("Blood Pressure Value(0-125)", min_value=0, max_value=125, step=10, format='%d')
                    Insulin = st.number_input("Insulin Level(0-850)", min_value=0, max_value=850, step=30)
                    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function Value(0.078 - 2.5)", min_value=0.078, max_value=2.5, format='%.1f')
    
                with col2:
                    Glucose = st.number_input("Glucose Level(0-200)", min_value=0, max_value=200, step=10, format='%d')
                    SkinThickness = st.number_input("Skin Thickness Value in mm(0-9.9)", min_value=0.0, max_value=10.0, step=0.2, format='%.1f')
                    BMI = st.number_input("BMI Value(0-40)", min_value=0, max_value=40, step=2)
                    Age = st.number_input("Age of the Person(21-85)", min_value=21, max_value=85)
    
                # Submit button in the form
                submitted = st.form_submit_button("Diabetes test Result")

            # Perform the prediction and show the result
            if submitted:
                diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
                st.success(diagnosis)



      
    elif selected == "Heart Disease Prediction":
            st.title("Heart Disease Prediction Web App")
        
            diagnosis = ''

            # Create a form for user input
            with st.form("heart_disease_form"):
                col1, col2, col3 = st.columns(3)
    
                with col1:
                    age = st.number_input("Age(29-77)" , min_value=29, max_value=77 , step=2 )
                    trestbps = st.number_input("Resting Blood Pressure(94-200)", min_value = 94 , max_value = 200 ,step =5)
                    restecg = st.number_input("Resting Electrocardiographic Results" , min_value = 0 , max_value = 2 , step=1)
                    oldpeak = st.number_input("ST Depression Induced by Exercise(0-6)",min_value = 0 , max_value = 6 , step=1)
                    thal = st.number_input("thal: 0 = normal; 1 = fixed defect; 2 = reversible defect" , min_value=0 , max_value=2)
        
                with col2:
                    sex = st.number_input("Sex" , min_value = 0 , max_value = 1 , step=1)
                    chol = st.number_input("Serum Cholesterol in mg/dl(130-550",min_value = 130 , max_value = 550 , step = 10)
                    thalach = st.number_input("Maximum Heart Rate Achieved(70 - 200)" , min_value = 70 , max_value = 200 , step = 5)
                    slope = st.number_input("Slope of the Peak Exercise ST Segment(0-2)" , min_value = 0 , max_value = 2 , step=1)
        
                with col3:
                    cp = st.number_input("Chest Pain Types" ,min_value = 0 , max_value = 3 , step=1)
                    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl(0<120 , 1>120)",min_value = 0 , max_value = 1 , step=1)
                    exang = st.number_input("Exercise Induced Angina" , min_value = 0 , max_value = 1 , step=1)
                    ca = st.number_input("Major Vessels Colored by Fluoroscopy",min_value = 0 , max_value=4 , step=1)
    
                # Submit button in the form
                submitted = st.form_submit_button("Heart Disease Test Result")

            # Perform the prediction and show the result
            if submitted:
                diagnosis = heart_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
            st.success(diagnosis)
            
            
            
            
    elif selected == "Parkinsons Disease Prediction":
            st.title("Parkinsons Disease Prediction Web App")
            
            diagnosis = ''

            # Create a form for user input
            with st.form("parkinsons_disease_form"):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    fo = st.number_input("MDVP:Fo(Hz)(88-260)" , min_value = 85 , max_value=260 , step=10)
                    Jitter_Abs = st.number_input("MDVP:Jitter(Abs)(0-0.01" , min_value = 0.0 , max_value = 0.01 , step=0.001)
                    DDP = st.number_input("Jitter:DDP")
                    APQ3 = st.number_input("Shimmer:APQ3")
                    spread1 = st.number_input("spread1")
                    
                with col2:
                    fhi = st.number_input("MDVP:Fhi(Hz)(100-600)", min_value = 100, max_value = 600 , step=20)
                    D2 = st.number_input("D2")
                    Shimmer = st.number_input("MDVP:Shimmer")
                    APQ5 = st.number_input("Shimmer:APQ5")
                    NHR = st.number_input("NHR")
                    DFA = st.number_input("DFA")
                   

                with col3:
                    flo = st.number_input("MDVP:Flo(Hz)(65-240)" , min_value = 65 , max_value = 240 , step=10)
                    RAP = st.number_input("MDVP:RAP")
                    Shimmer_dB = st.number_input("MDVP:Shimmer(dB)")
                    APQ = st.number_input("MDVP:APQ")
                    HNR = st.number_input("HNR")
                    spread2 = st.number_input("spread2")
                   

                with col4:
                    Jitter_percent = st.number_input("MDVP:Jitter(%)(0-1)" , min_value =0.0 , max_value = 1.0 , step=0.1 )
                    PPQ = st.number_input("MDVP:PPQ")
                    PPE = st.number_input("PPE")
                    DDA = st.number_input("Shimmer:DDA")
                    RPDE = st.number_input("RPDE")
            

                # Submit button in the form
                submitted = st.form_submit_button("Parkinson's Disease Test Result")
            # Perform the prediction and show the result
            if submitted:
                diagnosis = park_prediction([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
            st.success(diagnosis)
            
            
            
if __name__ == '__main__':
    main()
