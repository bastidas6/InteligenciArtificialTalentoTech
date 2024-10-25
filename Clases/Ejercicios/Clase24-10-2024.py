
#Tecnica OneHotEncoding
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = { 'asistencia_clases': ['baja', 'media', 'alta', 'media', 'alta'],
        'horas_estudio':[3,8,10,4,6],
         'Aprobado': ['Clase 1', 'Clase 2', 'Clase 2', 'Clase 1', 'Clase 2']
}

df2 = pd.DataFrame(data, columns = ['asistencia_clases', 'horas_estudio', 'Aprobado'])
print(df2)

enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(df2[['asistencia_clases']]).toarray(), columns=enc.categories_[0])
df3 = df2.join(enc_df)
print(df3)


#Tecnica de GetDummies, es mas sencillo que el OneHotEncoding, se ejecuta una sola tecnico, no se tienen que ejecutar varias
#El metodo de getdummies elimina la columna que esta codificando, o sea la columna original
#Drop_first = True: me elimina la primera columna ficticia
df = pd.get_dummies(df2, prefix="asistencia", columns=['asistencia_clases'])
print(df)


#Escalamiento de carcteristicas
# Normalización
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
horas_esc = min_max_scaler.fit_transform(df[["horas_estudio"]])
print(horas_esc)

print(df)

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
horas_esc = std_scaler.fit_transform(df[["horas_estudio"]])
print(horas_esc)