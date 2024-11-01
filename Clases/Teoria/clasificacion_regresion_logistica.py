#Clasificacion
#Entra en los algoritmos de aprendizaje supervisado y se busca predecir una variable discreta (tipo entero, 1 o 0, si o no, true and false)

#Regresión logistica: Modela la probabilidad de un elemento dependiente de ciertas variables independientes
#Funcion sigmoide => P = 1/(1 + e^(-y)), donde y => [0] + [1]X1 + [2]X2 + ... + [n]Xn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

#cargamos la data y visualizamos las primeras 5 filas
df = pd.read_csv("Datasets/diabetes.csv")
print(df.head(5))

# seleccionamos nivel de glucosa para X y diabetes para y
X = df.iloc[:, [6]]
y = df.iloc[:, 8].values

# se separan los datos en conjunto de entrenamiento y pruebas
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 47)

plt.figure(figsize=(4,3))
plt.scatter(X_train, y_train,  color='red',alpha=0.4,label=r'datos entrenamiento')
plt.legend(loc='lower right')
plt.xlabel(r'HbA1c_level')
plt.ylabel(r'Diabetes')
plt.show()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000) 
#Ajustando el modelo a los conjuntos de entrenamiento y prueba 
lr.fit(X_train, y_train)

#Puntuación de exactitud del modelo de regresión logística 
print("La exactitud del modelo es: ", lr.score(X_test, y_test))

W0 = lr.intercept_[0]
W1 = lr.coef_[0][0]
print(W0, W1)


y_pred = lr.predict(X_test)

sigmoide = 1 / (1 + np.exp(-(W0 + W1 * X_test)))  # Función sigmoide

# Graficar la función logit
plt.scatter(X_test, y_pred,  color='red',alpha=0.4,label='datos de prueba')
plt.plot(X_test, sigmoide, label="Función Sigmoide", color='blue')
plt.xlabel("Entrada (X)")
plt.ylabel("Probabilidad")
plt.title("Función Sigmoide estimada por la Regresión Logística")
plt.legend()
plt.grid()
plt.show()