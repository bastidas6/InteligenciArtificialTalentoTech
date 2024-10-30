import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv')
print(df.head(), "Tamaño: ", df.shape)

# eliminamos la columna index
df.drop(columns=['symboling','normalized-losses', 'make', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'peak-rpm','city-mpg', 'highway-mpg', 'diesel', 'gas'], inplace=True)
print(df.head())

# obtenemos engine-size, curb-weight y horsepower
X = df.iloc[:,[4,7,12]]
Y = df.iloc[:,13]

print("Valores de X: ", X, "Valores de Y: ", Y)


X_ent, X_pru, y_ent, y_pru = train_test_split(X, Y, test_size=0.2, random_state=1)

reg = LinearRegression()
reg.fit(X_ent, y_ent)

y_pred = reg.predict(X_pru)

exactitud = r2_score(y_pru, y_pred)
print(f'El R2 ES: {round(exactitud, 2)}')


