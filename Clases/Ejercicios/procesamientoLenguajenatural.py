#Interpretar lenguaje natural: Un sistema que etsudia una entrada del ser humano y devuelve una salida en el mismo idioma.
#Se debe generar la comunicacion con la persona, ya sea por texto o por medio de la voz.
#El NLP es una parte de la IA

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re

df = pd.read_csv("Datasets/reviews.csv")
print(df.sample(10))

corpus = pd.Series(df.Review.tolist()).astype(str)
print(corpus[727])

def limpiar_texto(corpus, ls_permitir):   
    corpus_limpio = []
    for fila in corpus:
        qs = []
        for palabra in fila.split():
            if palabra not in ls_permitir:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=palabra)
                p1 = p1.lower()
                qs.append(p1)
            else: 
                qs.append(palabra)        
        corpus_limpio.append(' '.join(qs))
    return pd.Series(corpus_limpio)

def remover_palabras_vacias(corpus):
    palabras = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
    stop = set(stopwords.words('english'))
    for palabra in palabras:
        stop.remove(palabra)
    corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    return corpus

def lematizar(corpus):
    lem = WordNetLemmatizer()
    corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    return corpus

def stem(corpus, stem_type = None):
    if stem_type == 'snowball':
        stemmer = SnowballStemmer(language = 'english')
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    else :
        stemmer = PorterStemmer()
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    return corpus

palabras_comunes = ['U.S.A', 'Mr.', 'Mrs.', 'D.C.']
corpus = limpiar_texto(corpus, palabras_comunes)
corpus = remover_palabras_vacias(corpus)
#corpus = [[x for x in ] for x in corpus]
corpus = [[x for x in oraciones] for oraciones in corpus]
corpus = lematizar(corpus)
corpus = stem(corpus, 'snowball')    
corpus = [' '.join(x) for x in corpus]    

print(corpus[727])