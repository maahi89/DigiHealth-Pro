// this file contains the frontend code of DigiHealth Pro

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 00:29:19 2023

@author: MAHITHA_GUDA
"""

import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

#loading the saved models
#diabetes model
diabetes_model, scaler1 = pickle.load(open('C:/Users/rohan/Desktop/multiple_disease_prediction_system/project_model/diabetes_model.pkl','rb'))
#heart disease model
heart_disease_model, scaler2 = pickle.load(open('C:/Users/rohan/Desktop/multiple_disease_prediction_system/project_model/heart_disease_model.pkl','rb'))
#parkinson model
parkinson_model, scaler3 = pickle.load(open('C:/Users/rohan/Desktop/multiple_disease_prediction_system/project_model/parkinson_model.pkl','rb'))

#creating the side bar UI for navigation

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           
                           icons = ['activity',
                                    'heart',
                                    'person'],
                           
                           default_index = 0)
    
# Diabetes prediction page
if (selected == 'Diabetes Prediction'):
    
    #page title
    st.title('Diabetes Prediction') 
    
    #getting the input data from the user
    #columns for the input field
    col1, col2 = st.columns(2)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose level')

    with col1:
        BloodPressure = st.text_input('Blood Pressure')
        
    with col2:
        SkinThickness = st.text_input('Skin Thickness Value')

    with col1:
        Insulin = st.text_input('Insulin level')

    with col2:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')
         
    #code for Prediction of the disease
    diab_diagnosis = ''
    
    # creating a button for taking the input and predicting the ouput
    if st.button('Diabetes Test Result'):
        
        input_data_1 = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        
        # changing the input_data into numpy array
        input_data_1_as_numpy_array = np.asarray(input_data_1)
        input_data_1_reshaped = input_data_1_as_numpy_array.reshape(1, -1)
        
        # standardize the input_data
        std_data_1 = scaler1.transform(input_data_1_reshaped)
        
        diab_prediction = diabetes_model.predict(std_data_1)
        
        if (diab_prediction[0] == 0):
            diab_diagnosis = 'The person is NOT HAVING Diabetes'
        
        else:
            diab_diagnosis = 'The person is HAVING Diabetes'
        
        st.success(diab_diagnosis)
        
    
    
    
    
if (selected == 'Heart Disease Prediction'):
    
    #page title
    st.title('Heart Disease Prediction ') 
   
    #getting the input data from the user
    #columns for the input field
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age of the Person')
        
    with col2:
        sex = st.text_input('Sex of the Person: 1=M & 0=F ')

    with col3:#4 values are there for chest pain 0,1,2,3
        cp = st.text_input('Type of Chest Pain')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestrol in mg/dl')

    with col3:#FBS = FASTING BLOOD SUGAR
        fbs = st.text_input('FBS>120 mg/dl: 1=T & 0=F')
    
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Agina: 1=Y & 0=N')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
         
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
         
    #code for Prediction of the disease
    heart_diagnosis = ''
    
    #creating a button for prediction
    if st.button('Heart Disease Test Result'):
        input_data_2 = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
        
        # changing the input_data into numpy array
        input_data_2_as_numpy_array = np.asarray(input_data_2)
        input_data_2_reshaped = input_data_2_as_numpy_array.reshape(1, -1)
        
        # standardize the input_data
        std_data_2 = scaler2.transform(input_data_2_reshaped)
        
        heart_disease_model_prediction = heart_disease_model.predict(std_data_2)
        
        if (heart_disease_model_prediction[0] == 0):
            heart_diagnosis = 'The person is  NOT HAVING heart disease.'
        else:
            heart_diagnosis = 'The person is HAVING heart disease'
    
    st.success(heart_diagnosis)
    
if (selected == 'Parkinsons Prediction'):
    
    #page title
    st.title('Parkinsons Prediction') 
    
    #getting the input data from the user
    #columns for the input field
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fo = st.text_input('MDVP Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP Jitter(%)')

    with col1:
        Jitter_Abs = st.text_input('MDVP Jitter (Abs)')

    with col2:
        RAP = st.text_input('MDVP RAP')
    
    with col3:
        PPQ = st.text_input('MDVP PPQ')

    with col4:
        DDP = st.text_input('Jitter DDP')
        
    with col1:
        Shimmer = st.text_input('MDVP Shimmer')
        
    with col2:
        Shimmer_dB = st.text_input('MDVP Shimmer (dB)')
    
    with col3:
        APQ3 = st.text_input('Shimmer APQ3')
         
    with col4:
        APQ5 = st.text_input('Shimmer APQ5')
        
    with col1:
        APQ = st.text_input('MDVA APQ')
         
    with col2:
        DDA = st.text_input('Shimmer DDA')
        
    with col3:
        NHR = st.text_input('NHR')
    
    with col4:
        HNR = st.text_input('HNR')
    
    with col1:
        RPDE = st.text_input('RPDE')
    
    with col2:
        DFA = st.text_input('DFA')
    
    with col3:
        spread1 = st.text_input('spread1')

    with col4:
        spread2 = st.text_input('spread2')
    
    with col1:
        D2 = st.text_input('D2')  
    
    with col2:
        PPE = st.text_input('PPE')
    
    #code for Prediction of the disease
    parkinson_diagnosis = ''
    
    #creating a button for prediction
    if st.button('Parkinsons Disease Test Result'):
        input_data_3 = (fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE)
        
        # changing the input_data into numpy array
        input_data_3_as_numpy_array = np.asarray(input_data_3)
        input_data_3_reshaped = input_data_3_as_numpy_array.reshape(1, -1)
        
        # standardize the input_data
        std_data_3 = scaler3.transform(input_data_3_reshaped)
        
        parkinson_model_prediction = parkinson_model.predict(std_data_3)
        
        if (parkinson_model_prediction[0] == 0):
            parkinson_diagnosis = 'The person is  NOT HAVING parkinson disease.'
        else:
            parkinson_diagnosis = 'The person is HAVING parkinson disease'
    
    st.success(parkinson_diagnosis)
    
    

    
