import pandas as pd

df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv')
print(df.head(), "Tamaño: ", df.shape, "Columnas: ", df.dtypes)


#Vamos a ver si puedo aplicar regresion lineal con el grafico de mapa de calor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cols = ['wheel-base','curb-weight','engine-size','horsepower','city-mpg', 'price']

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size':15},
                yticklabels=cols,
                xticklabels=cols
                )
plt.show()


#Vamos a ver la dispercion de los datos, para halalr las variqables mas significativas
sns.set(style='ticks')
sns.pairplot(df[cols])
plt.show()


#Regresion lineal
# 1. Decidir cuales son las x y cuales son la y
X = df[['engine-size']].values
Y = df['price'].values
#print("Valores de X: ",X,"Valores de Y: ", Y)

# 2. Separacion del conjunto de datos
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=1)

# 3. Seleccionamos el algortimo de regresion lineal y se entrena el modelo
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

# 4. Hallar intercepto y pendiente
  # 4.1 Parametro: Se generan dentro del mismo modelo, en funcion del mismo modelo (coeficientes)
  # 4.2 Hiper-parametro: Son valores externos, que le configuro al modelo para que en base a esos valores que yo le di haga la tarea que se le encomienda y sirve para ajustar mas el entrenamiento.
print("Intercepto: ", round(reg.intercept_,2), "Pendiente: ", round(reg.coef_[0]))


# 5. Realizar prediccion
y_pred = reg.predict(x_test)
print(y_pred)
#Se aplica ecuacion de la recta
print(reg.intercept_ + reg.coef_[0]*x_test[0])

# 6. Aplicamos metrica de evaluacion para saber si el modelo me quedo bien
from sklearn.metrics import r2_score
print("Coeficiente r2: ",round(r2_score(y_test, y_pred),2)) #En este caso nos dio 0.68, no es lo esperado pero tiene algo de bueno

# 7. Hacemos el grafico
plt.scatter(x_test, y_test, color='blue', label= "Datos de prueba")
plt.scatter(x_train, y_train, color='red', label= "Datos de entrenamiento")
plt.plot(x_test, y_pred, color='blue', label = 'Prediccion')
plt.legend()
plt.xlabel("Tamaño del motor")
plt.ylabel("Precio")
plt.show()