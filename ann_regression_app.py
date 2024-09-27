# importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st
import dill

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# loading all models
with open("oher.pkl", "rb") as file:
    ohe = pkl.load(file)
with open("scr.pkl", "rb") as file:
    sc = pkl.load(file)
with open("gender_encode.dill", "rb") as file:
    ge = dill.load(file)

model = load_model("annr.h5")


st.title("Estimated Salary prediction")

CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography', ohe.categories_[0])
Gender = st.selectbox('Gender', ['Male', 'Female'])
Age = st.slider('Age', 18, 92)
Tenure = st.slider("Tenure", 0, 10)
Balance = st.number_input('Balance')
NumOfProducts = st.number_input('Number Of Products', 1, 4)
HasCrCard = st.selectbox('Has Credit Card', [0, 1])
IsActiveMember = st.selectbox('Is Active Member', [0, 1])
Exited = st.selectbox('Exited',[0,1])

idata = {
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Gender': Gender,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'Exited': Exited
}

df1 = pd.DataFrame([idata.values()], columns=idata.keys())


# Gender encoding
def preprocess(id):
    id["Gender"] = id["Gender"].map({"Male": 1, "Female": 0})  # Simplified gender encoding
    id1 = pd.DataFrame(ohe.transform(id["Geography"].values.reshape(-1, 1)).toarray(), 
                       columns=ohe.get_feature_names_out(['Geography']))
    id.drop("Geography", inplace=True, axis=1)
    id = pd.concat([id, id1], axis=1)
    id = sc.transform(id)
    return id



# Ensuring consistent input shape for prediction
df1 = preprocess(df1)
df1 = np.array(df1).reshape(1, -1)  # Reshaping for batch consistency

# Predict outside any loop and avoid retracing
pred = model.predict(df1)
st.write(f"Estimated Salary :{pred[0][0]:.2f}")
