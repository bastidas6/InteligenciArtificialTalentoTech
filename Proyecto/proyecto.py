import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#----------------------------------------------------MISIÓN 1 ---------------------------------------------

                               #CARGA Y EXPLORACION DEL CONJUNTO DE DATOS

#Cargamos el dataset
df = pd.read_csv("Proyecto/Homicidios_Colombia.csv", delimiter=";")

#Miramos la cantidad de filas y de columnas del dataset
print("El número de filas y de columnas del dataset es:\n", df.shape)

#TIPOS DE DATOS
# Identificar los tipos de datos de cada columna
print("Tipos de datos de cada columna:\n", df.dtypes)

#CANTIDAD DE DATOS FALTANTES
# Detectar la cantidad de datos faltantes en cada columna
faltantes_por_columna = df.isnull().sum()
print("Cantidad de datos faltantes en cada columna:\n", faltantes_por_columna)

#VERIFICACIÓN DE REGLAS DE RANGO Y DOMINIO

