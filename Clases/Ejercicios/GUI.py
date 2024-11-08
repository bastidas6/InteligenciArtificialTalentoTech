#Hacemos interfaz grafica
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Programa para predecir cancer")
st.write("Debes ingresar dos datos para obtener una prediccion")

#Cargamos el modelo
modelo = joblib.load("modelo_svm_final.pkl")

#Creamos los campos de la GUI
radio = st.number_input("Radio medio", min_value=0.1, max_value=100.0, step=0.1)
textura = st.number_input("Textura media", min_value=0.1, max_value=100.0, step=0.1)

if (st.button("Predecir")):
    datos = np.array([[radio, textura]])
    prediccion = modelo.predict(datos)
    if prediccion[0] == 1:
        res = "La prediccion es: tumor maligno"
    else:
        res = "La prediccion es tumor benigno"
    st.write(f"{res}")