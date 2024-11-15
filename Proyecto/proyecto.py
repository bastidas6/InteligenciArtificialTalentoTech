import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#----------------------------------------------------MISIÓN 1 ---------------------------------------------

                               #CARGA Y EXPLORACION DEL CONJUNTO DE DATOS

#Cargamos el dataset
df = pd.read_csv("Proyecto/Homicidios_Colombia.csv", delimiter=";")
faltantes_por_columna = df.isnull().sum()
print("Cantidad de datos faltantes iniciales en cada columna:\n", faltantes_por_columna)


#Miramos la cantidad de filas y de columnas del dataset
print("El número de filas y de columnas del dataset es:\n", df.shape)

#TIPOS DE DATOS
# Identificar los tipos de datos de cada columna
print("Tipos de datos de cada columna:\n", df.dtypes)
print(df.head(5))

#Categorizamos la columna sexo
d1 = {'Hombre': 1, 'Mujer': 2, 'Indeterminado':3}
df['Sexo de la victima']=df['Sexo de la victima'].map(d1)


#Categorizamos la columna dia del hecho
d2 = {'Lunes': 1,'lunes': 1, 'Martes': 2,'martes': 2 ,'Miércoles': 3,'miércoles': 3, 'Jueves': 4, 'jueves': 4, 'Viernes': 5, 'viernes': 5, 'Sábado': 6, 'sábado': 6,'Domingo': 7, 'domingo': 7, 'Sin Información': 8}
df['Dia del hecho']=df['Dia del hecho'].map(d2)


#Categorizamos la columna Grupo de edad de la victima
d3 = {'(00 a 04)': 1,'(05 a 09)': 2,'(10 a 14)': 3, '(15 a 17)': 4, '(18 a 19)': 5, '(20 a 24)': 6, '(25 a 29)': 7, '(30 a 34)': 8, '(35 a 39)': 9, '(40 a 44)': 10, '(45 a 49)': 11, '(50 a 54)': 12,'(55 a 59)': 13, '(60 a 64)': 14, '(65 a 69)': 15, '(70 a 74)': 16, '(75 a 79)': 17, '(80 y más)': 18, 'Por determinar': 19}
df['Grupo de edad de la victima']=df['Grupo de edad de la victima'].map(d3)

#Categorizamos la columna Grupo de mes del hecho
d4 = {'Enero': 1, 'enero': 1,'Febrero': 2,'febrero': 2, 'Marzo': 3, 'marzo': 3,'Abril': 4,'abril': 4, 'Mayo': 5,  'mayo': 5, 'Junio': 6, 'junio': 6, 'Julio': 7, 'julio': 7,'Agosto': 8,'agosto': 8 ,'Septiembre': 9, 'septiembre': 9,'Octubre': 10,'octubre': 10 ,'Noviembre': 11,'noviembre': 11 ,'Diciembre': 12, 'diciembre': 12, 'Sin Información':13}
df['Mes del hecho']=df['Mes del hecho'].map(d4)


#Categorizamos la columna Departamento del hecho DANE
d5 = {'Antioquia': 1, 'Amazonas': 2, 'Arauca': 3, 'Archipiélago de San Andrés, Providencia y Santa Catalina': 4, 'Atlántico': 5, 'Bogotá, D.C.': 6, 'Bolívar': 7, 'Boyacá': 8, 'Córdoba': 9, 'Caldas': 10, 'Caquetá': 11, 'Casanare': 12, 'Cauca': 13, 'Cesar': 14, 'Chocó': 15, 'Cundinamarca': 16, 'Guainía': 17, 'Guaviare': 18, 'Huila': 19, 'La Guajira': 20, 'Magdalena': 21, 'Meta': 22, 'Nariño': 23, 'Norte de Santander': 24, 'Putumayo': 25, 'Quindío': 26, 'Risaralda': 27, 'Santander': 28, 'Sucre': 29, 'Tolima':30, 'Valle del Cauca': 31, 'Vaupés': 32, 'Vichada': 33, 'Sin Información': 34, 'Sin información': 34}
df['Departamento del hecho DANE']=df['Departamento del hecho DANE'].map(d5)


#Categorizamos la columna dependiente Magnitud del homicidio
d6 = {'Alto Riesgo': 1, 'Bajo Riesgo': 2}
df['Magnitud del homicidio']=df['Magnitud del homicidio'].map(d6)
print(df.head(5))


#ELIMINAMOS COLUMNAS
df.drop(columns=['Estado', 'Lesion fatal de causa externa', 'Municipio del hecho DANE', 'ID'], inplace=True)
print(df.head(5))
print(df.describe())

#CANTIDAD DE DATOS FALTANTES
# Detectar la cantidad de datos faltantes en cada columna
faltantes_por_columna = df.isnull().sum()
print("Cantidad de datos faltantes en cada columna:\n", faltantes_por_columna)

#CALCULO DE LA CORRELACION
print("La correlación es: ", df[["Año del hecho","Sexo de la victima"]].corr())
print(df.head(5))



