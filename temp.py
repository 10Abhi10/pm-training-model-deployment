# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import joblib
# Libraries for data exploration and manipulation
import pandas as pd
df_v1 = pd.read_csv(r'C:\Users\VZ969\Downloads\Project_Data\DMD_Intake_12_MON_original - Copy 1.csv')

df_v1.drop('PART_ID', axis = 1, inplace = True)

df_v1.columns = ['LT', 'Category', 'HITS', 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5',
                'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12']

# Validation dataset
df_val = df_v1.sample(1000)

# Indices of the validation set
index_variable = list(df_val.index)

# Training dataset
df_input = df_v1.drop(index_variable, axis = 0)

# models of regression
from pycaret.regression import *
s = setup(df_input, target = 'Month_12', session_id = 123)

# Comparing different models
best_model = compare_models()

# Prediction on test set (30% selected by Pycaret)
df_predict = predict_model(best_model)

# training & evaluating the performance of the model using cross-validation
et = create_model('et')

prediction_val = predict_model(et, data = df_val)

col = df_v1.columns

# removing the label Month_12 from the list of columns
col = col [:-1]

# dropping the Month_1 data and shifting the remaining data to left for predicting for 13th Month
df_final = df_v1.drop('Month_1', axis = 1)
df_final.columns = col

prediction_final = predict_model(et, data = df_final)

final_et = finalize_model(et)

unseen_prediction = predict_model(final_et, data = df_final)

save_model(final_et, 'C:/Users/VZ969/.spyder-py3/final_model')

model = joblib.load('C:/Users/VZ969/.spyder-py3/final_model.pkl')

st.title('Demand Prediction Model Deployment')
st.write('This is a Streamlit web app to demonstrate deploying a Demand Prediction model.')

# Function to predict using the loaded model
def predict(input_features):
    prediction = model.predict(input_features)
    return prediction

prediction_x = predict(df_final)

st.write('Predicted Output:', prediction_x[0])