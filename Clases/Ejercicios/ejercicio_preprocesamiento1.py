#1. Analisis exploratorio de datos: Hacer un analisis superficial de los datos, no vamos hacer nada con los datos, simplemente es un diagnostico. 
#2. Preprocesamiento de datos:

#Importar librerias
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Cargar archivo
df = pd.read_csv("Datasets/titanic.csv", delimiter=",")
#Imprimir las 3 primeras filas
print("\nPrimeras 3 filas\n",df.head(3))

#Imprimir la forma del data frame}
print("\nTamaño\n",df.shape)

#Imprimir tipos de datos de las columnas
print("\nTipos de datos\n", df.dtypes)

#Datos estadisticos de la tabla
print("\n Valores estadisticos\n", df.describe())

#Graficar edad vs miedo
sns.regplot(x = 'Age', y = 'Fare', data = df)
plt.show()
#sns.barplot(x = 'Age', y = 'Survived', data = df)
#plt.show()

#Hallar correlacion entre dos variables > 0, directamente proporcional, < 0 inversamente proporcional, =0 no hay relacions
print(df[["Age","Fare"]].corr())

#Resumen de datos
#Agrupar por "Pclass" y hallar la media de la edad y el fare
df_grupo = df[['Pclass', 'Age', 'Fare']]
df_res = df_grupo.groupby(['Pclass'], as_index=False).mean()
print(df_res)

#Preprocesamiento

#Obtener valores nulos
#1. Una opcion es eliminar las filas con valores nulos, pero aplica si son muchos datos
#2. Imputacion, es aplicar el metodo de imputar para reemplazar los valores
print("\nValores nulos\n", df.isna().sum())

from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')
df['Age'] = imp.fit_transform(df[['Age']])
print("\nValores nulos\n", df.isna().sum())

#Eliminar fila donde el fare es invalido
#Con el inplace = True, se elimina la fila y se refleja en el dataset
df.dropna(axis=0, how='any', subset=['Fare'], inplace=True)
print("\nValores nulos\n", df.isna().sum())


#Eliminar columnas innecesarias con drop
print(df.columns)
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
print("\nDataFrame con columnas eliminadas\n", df.head(3))


#Se transforman y/o se categorizan las variables categoricas

#Se transforma columna Sex
#print(df['Sex'].unique())  #Conocer posibles valores que puede tener la columna sex
diccionario_sex = {'male':1, 'female':2}
df['Sex'] = df['Sex'].map(diccionario_sex)
print("\nDataFrame con la columna sex transformada\n", df.head(3))

#Se transofrma columnas Embarked
#print(df['Embarked'].unique())  #Conocer los posibles valores que puede tener la columna Embarked
#diccionario_embarked = {'Q':1, 'S':2, 'C':3}
#df['Embarked'] = df['Embarked'].map(diccionario_embarked)
#print("\nDataFrame con la columna Embarked transofrmada\n", df.head(3))


valores_posibles_embarked = df['Embarked'].unique()
diccionario_embarked = {}
contador_embarked = 1
for item in valores_posibles_embarked:
    diccionario_embarked[item] = contador_embarked
    contador_embarked +=1

print(diccionario_embarked)

df['Embarked'] = df['Embarked'].map(diccionario_embarked)
print("\nDataFrame con la columna Embarked transofrmada\n", df.head(3))