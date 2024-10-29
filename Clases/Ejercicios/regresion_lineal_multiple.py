import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/insurance.csv")
print(df)

# codificamos la variable sex
d1 = {'male': 1, 'female':0}
df['sex'] = df['sex'].map(d1)

# codificamos la variable smoker
d2 = {'yes': 1, 'no':0}
df['smoker'] = df['smoker'].map(d2)

# codificamos la variable region
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# eliminamos la columna index
df.drop(columns='index', inplace=True)
print(df)

import seaborn as sb
import matplotlib.pyplot as pl
cm = np.corrcoef(df.values.astype(np.float64).T)
sb.set(rc = {'figure.figsize':(10,8)})
pl.figure(figsize=(8,6))
hm = sb.heatmap(cm, cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size':15},
                yticklabels=df.columns,
                xticklabels=df.columns
                )
pl.show()

# obtenemos age, bmi y smoker
X = df.iloc[:,[0,2,4]]
y = df.iloc[:,5]


X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=1)

reg = LinearRegression()
reg.fit(X_ent, y_ent)

y_pred = reg.predict(X_pru)

exactitud = r2_score(y_pru, y_pred)
print(f'La exactitud es: {round(exactitud, 2)}')

prueba = np.array([[36, 27, 0]])
print("El cobro del seguro es: ", round(reg.predict(prueba)[0],2))