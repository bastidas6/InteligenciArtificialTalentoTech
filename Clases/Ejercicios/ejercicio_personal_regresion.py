import pandas as pd
import numpy as np

df = pd.read_csv("Datasets/titanic.csv", delimiter=",")
print(df.tail())

#df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
#print(df.tail())

df.drop(columns=['PassengerId', 'Name', 'Pclass', 'SibSp', 'Parch', 'Ticket', 'Sex', 'Cabin', 'Embarked', 'Survived'], inplace=True)
print(df)

df.dropna(subset=['Fare', 'Age'], inplace=True)



X = df[['Fare']].values 
y = df.Age

from sklearn.model_selection import train_test_split
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.3, random_state=1) #El 20% de los datos se van para el conjunto de pruebas

from sklearn.linear_model import LinearRegression
reg_lin = LinearRegression()
print(reg_lin.fit(X_ent, y_ent))
print("Intercepto con el eje X: ", reg_lin.intercept_, "Coeficiente[0]: ", reg_lin.coef_[0])

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(X_ent, y_ent, color='blue', label='Datos de entrenamiento')
plt.plot(X_ent, reg_lin.predict(X_ent), color='red', label='Recta de mejor ajuste')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.title('Regresión lineal')
plt.show()