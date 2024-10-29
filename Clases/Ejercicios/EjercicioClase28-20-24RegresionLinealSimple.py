#Modelos de Regresion

#Aprendizaje supervisado: COnjunto de algortimos de la IA: Machine Learning
     #Aprendizaje supervisado: Tiene variable objetivo (etiqueta), este es mas preciso y se obtienen mejores resultados.
     #Aprendizaje No Supervisado: No tiene variable objetivo, solo contamos con las variable descriptivas

#Aprendizaje supervisado
     #Clasificacion: Se produce una variable discreta, de tipo entero y se asoscia con una clase
     #Regresion: Se va a predecir una variable continua(tipo real)
          #Regresion Lineal: Los datos dibujasn una linea recta.
                 #Regresion lineal simple: Se caracteriza por que solo tiene una variable independiente para predecir la variable dependiente, se modela con la ecuación de la linea recta y=W0 + W1X
                 #Regresion lineal multiple: Se tienen en cuenta dos o mas caracteristicas y la variable objetivo 
          #Regresion Polinomica: Se puede transformar y se puede tratar como una regresion lineal multiple


import pandas as pd

df = pd.read_csv("Datasets/Salary_Data.csv")
print(df.tail())

# Caracteristicas o variables independientes
X = df[['YearsExperience']].values #Esto devuelvo toda la columna de YearsExperience.

# Variable dependiente u objetivo
y = df.Salary #Esto me devuelve toda la columna de Salary


from sklearn.model_selection import train_test_split
# separación de los datos en conjuntos de entrenamiento y pruebas
X_entren, X_prueba, y_entren, y_prueba = train_test_split(X, y, test_size=0.20, random_state=0) #El 20% de los datos se van para datos de prueba.
print("X de entrenamiento:", X_entren.shape,"\n X de prueba: ", X_prueba.shape,"\nY de entrenamiento: ", y_entren.shape,"\n Y de prueba: ", y_prueba.shape)


from sklearn.linear_model import LinearRegression
reg_lin = LinearRegression()
print(reg_lin.fit(X_entren, y_entren))
print("Intercepto con el eje X: ", reg_lin.intercept_, "Coeficiente[0]: ", reg_lin.coef_[0])

predicciones = reg_lin.predict(X_prueba)
print(predicciones)

print("wo:", format(reg_lin.intercept_,".2f"))
print("w1:",format(reg_lin.coef_[0],".2f"))

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(X_entren, y_entren, color='blue', label='Datos de entrenamiento')
plt.plot(X_entren, reg_lin.predict(X_entren), color='red', label='Recta de mejor ajuste')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.title('Regresión lineal')
plt.show()

# Error absoluto medio
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_prueba, predicciones)
print(f'El Error absoluto medio es: {round(mae, 2)}')

#Error cuadrático medio
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_prueba, predicciones)
print(f'El Error cuadrático medio es: {round(mse, 2)}')

from sklearn.metrics import  r2_score
exactitud = r2_score(y_prueba, predicciones)
print(f'La exactitud es: {round(exactitud, 2)}')

valor_y = reg_lin.intercept_ + reg_lin.coef_[0]*9.7
print("El valor predictorio es: ", valor_y)




