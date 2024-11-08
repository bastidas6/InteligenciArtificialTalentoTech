#Vamos a predecir el cancer en funcion de dos variables predictoras(independientes y/o explicativas)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Datasets/Cancer.csv")
print(df.head())
#print("Las columnas son: ", df.columns)
print("Los valores de diagnosis es: ", df['diagnosis'].unique())

#Vamos a transofrmar(codificar o mapear) la columna diagnosis
diccionario_diagnosis = {"M": 1, "B": 0}
df["diagnosis"] = df["diagnosis"].map(diccionario_diagnosis)
#print(df.head())

#Eliminamos la columna sin nombre: Unnamed: 32
df.drop(columns=["Unnamed: 32"], inplace=True)
print(df.head())
#print(df.shape)

#Vamos a coger dos variables y las vamos a graficar en un plano bidimensional, cogemos radius_mean y texture_mean
X = df.iloc[0:100,[2, 3]]
Y = df["diagnosis"].iloc[:100]

#Separamos los datos en conjunto de entrenamiento y pruebas
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=47)

#Graficamos  --> Revisar por que no da
'''plt.figure(figsize=(6,4))
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c= "red", marker="s", label = "Malignos") #Graficamos solo los que tienen tumor maligno, M = 1
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c= "blue", marker="o", label = "Benignos") #Graficamos solo los que tienen tumor Benigno, M = 0
plt.xlabel("")
plt.ylabel("")
plt.legend(loc="best")
plt.show()'''

#Aplicamos algoritmo de regresion logistica
from sklearn.linear_model import LogisticRegression
rl = LogisticRegression()
modelo_rl = rl.fit(X_train, y_train) #Ya tenemos el modelo ajustado con los datos de entrenamiento

#Hacemos predccion con los datos de prueba
prediccion = modelo_rl.predict(X_test)


#Evaluamos el modelo de regresion logistica, escogemos la metrica favorita
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score #Exactitud, matriz de confusión y precision
print(confusion_matrix(prediccion, y_test))
print("La exactitud es: ", accuracy_score(prediccion, y_test))

# Entrenar el KNN con los mismos datos
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
modelo_knn = knn.fit(X_train, y_train)
#Evaluamos modelo
predicciones = modelo_knn.predict(X_test)
print(confusion_matrix(predicciones, y_test))
print("La exactitud es: ", accuracy_score(predicciones, y_test))


#Entrenar con bosque aleatorio
from sklearn.ensemble import RandomForestClassifier
rforest = RandomForestClassifier(criterion='gini', n_estimators= 100)
modelo_rf = rforest.fit(X_train, y_train)
predicciones_forest = modelo_rf.predict(X_test)
print(confusion_matrix(predicciones_forest, y_test))
print("La exactitud es: ", accuracy_score(predicciones_forest, y_test))


#Entrenamos con Nave Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
modelo_nb = nb.fit(X_train, y_train)
prediccion_nb = modelo_nb.predict(X_test)
print(confusion_matrix(prediccion_nb, y_test))
print("La exactitud es: ", accuracy_score(prediccion_nb, y_test))

#Entrenamos algoritmo SVM
from sklearn import svm
modelo_svm = svm.SVC(kernel='linear')
modelo1 = modelo_svm.fit(X_train, y_train)
prediccion_svm = modelo1.predict(X_test)


#Entrenamos con el modelo svm con gridseachcv
'''from sklearn.model_selection import GridSearchCV
from sklearn import svm
model = svm.SVC()
 
grilla = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.01, 0.25, 0.5, 0.75, 1.0],
    'kernel': ['rbf', 'poly']
}
 
busqueda = GridSearchCV(estimator=model, param_grid=grilla, cv=5, verbose=1)
print(busqueda.fit(X_train, y_train)) # Entrenar el modelo con diferentes combinaciones

mejor_modelo = busqueda.best_estimator_
print(mejor_modelo)
from sklearn import svm
modelo = svm .SVC(kernel='poly', C=0.01, gamma=0.01)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
from sklearn.metrics import accuracy_score
print("La exactitud es: ", accuracy_score(predicciones, y_test))'''

#Vamos a graficar una interfaz de usuario
#Streamlit --> Aplicaciones de front muy sencillas --> Instalamos StreamLit

#Exportamos el modelo --> Para que quede en el disco duro y no en memoria
import joblib
joblib.dump(modelo1, 'modelo_svm_final.pkl')

