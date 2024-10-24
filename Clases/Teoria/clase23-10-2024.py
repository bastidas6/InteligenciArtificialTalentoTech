#Variables cualitativas (Nominales, Ordinales y Binarias) y cuantitativas (Discretas y continuas)
#Datos estructurados: Tablas, JSON ---------- Datos no estructurados: Videos, Sonidos e Imagenes
#Preprocesamiento de datos: conjunto de tecnicas que aseguran la viabilidad y la eficiencia, dejar los datos de forma adecuada para luego pasarlos a un modelo de ML.
   #Limpieza de datos: Datos nulos (NaN), datos duplicados, datos mal escritos, datos atipicos, tipos de datos inconsistentes.
   #Transformacion de datos: Filtrar los datos, traduccion de datos cualitativos a numeros o estructuras matematicas, dejar los datos en una misma escala (normalizacion)
   #Reduccion de datos: Reduccion de dimensionalidad, quitar columnas innecesarias - PCA Algorithm

#Procesamiento de datos
   #Seleccionar la data y dividirlos en entrenamiento, datos para que el modelo aprenda y los datos de test es para evaluar si el modelo predice correctamente.
   #Procesar la data, seleccionar las columnas relevantes

#Se importan las librerias
import pandas as pd
import matplotlib.pyplot as plt


#Se crea el dataframe cargando la data
df = pd.read_csv("Archivos/dataset_car.csv", delimiter=";")

#Mostrar 5 filas de forma aleatoria
datos_aleatorios = df.sample(5)
print("\n5 datos aleatorios: \n", datos_aleatorios)

#Resumen estadistico del dataset
print("\nEste es el resumen estadistico: \n", df.describe())

#Numero de clases de la columna "class"
print("\nValores por cada clase de la columna class: \n\n", df['class'].value_counts())

#Tamaño del data set
print("\nEl tamaño del dataset es: \n\n", df.shape)

#Las columnas son
print("\n Las columnas son: \n", df.columns)

#Dibujar el numero de carros por clase
plt.figure(figsize=(6,4))
plt.bar(df['class'].unique(), list(df['class'].value_counts()))
plt.title('frecuencia variable objetivo')
plt.xlabel('Class')
plt.ylabel('Instancias')
#plt.show()


#Preprocesamiento
#Cambiar valores de la columna Buying ya que es una variable categorica

#Creamos diccionario con los posibles valores de la columnas
#print(df['Buying'].unique())
d1 = {'vhigh':0, 'high': 1, 'med':2, 'low':3}
d2 = {'vhigh':0, 'high': 1, 'med':2, 'low':3}
d3 = {'2':2, '3':3, '4':4, '5more':5}
d4 = {'2':0, '4':1, 'more':2}
d5 = {'small':0, 'med':1, 'big':2}
d6 = {'low':0, 'med':1, 'high': 2}

df['Buying'] = df['Buying'].map(d1)
df['Maintenance'] = df['Maintenance'].map(d2)
df['Doors']=df['Doors'].map(d3)
df['Person'] = df['Person'].map(d4)
df['lug_boot'] = df['lug_boot'].map(d5)
df['safety'] = df['safety'].map(d6)


#Para la columna class se va usar el algoritmo de LabelEncoder

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
df['class'] = enc.fit_transform(df['class'])

print("\nDataset con la columna buying mapeada: \n", df.sample(5))


#Identificar cuantos valores nulos hay en dataset
print("\nLos valores nulos son: \n", df.isna().sum())

##Division del dataset en datos de entrenamiento y datos de test
x = df.iloc[:,:-1].values #Traigame todas la filas y las columnas desde la 0 hasta la penultima
y = df.iloc[:,6].values #Traigame todas la filas y solo la ultima columna

print(x,"Y",y)

from sklearn.model_selection import train_test_split

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(x,y, test_size= 0.3, random_state=1) #test_size = el porcentaje que va hacer de prueba, random_state se usa para que se haga de forma aleatoria.