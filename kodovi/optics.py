# Biblioteka za citanje
import pandas as pd

# Biblioteka za rad sa OS
import os

# Biblioteka za cuvanje modela
from joblib import dump, load

# Biblioteka za klasterovanje
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score

# OPTICS model
def optics(skup, mime):
  # Za razlicite mere rastojanja
  for mera in ('manhattan', 'euclidean', 'cosine'):
    # Modelovanje
    klast = OPTICS(zmetric=mera, n_jobs=-1).fit(skup)
    
    # Cuvanje modela
    putanja = f'../modeli/optics/{mime}_optics_{mera[:3]}.joblib'
    os.makedirs(os.path.dirname(putanja), exist_ok=True)
    dump(klast, putanja)

# Pravljenje modela
def napravi():
  # Za svaki fajl iz projekta
  spisak = ['Pre', 'Disc_Early', 'Disc_Resp', 'Disc_AR']
  for i, desc in enumerate(spisak, start=3330561):
    # Ucitavanje skupa
    skup = pd.read_csv(f'../GSM{i}/GSM{i}_PBMC_{desc}_t.csv',
                       index_col=0)
    
    # Pravljenje modela
    optics(skup, f'GSM{i}')

  # Ucitavanje spojenog skupa
  skup = pd.read_csv(f'../GSM333056x/GSM333056x.csv',
                     index_col=0)

  # Pravljenje modela
  optics(skup, 'GSM333056x')

# Prikaz modela
def prikazi():
  # Za svaki skup
  for i in (*range(1, 5), 'x'):
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Za svaku meru
    for mera in ('manhattan', 'euclidean', 'cosine'):
      # Ucitavanje modela
      klast = load(f'../modeli/optics/GSM333056'
                   f'{i}_optics_{mera[:3]}.joblib')
      klasteri = klast.labels_
      
      # Racunanje senka koeficijenta
      skor = silhouette_score(skup, klasteri, metric=mera)\
             if len(set(klasteri)) > 1 else float('nan')

      # Pravljenje mape klastera
      mapa = {}
      for k in klasteri:
        mapa[k] = 1 if k not in mapa else mapa[k]+1

      # Stampanje rezultata
      print(f'{i}-{mera[:3]} ({skor:.2f}): {mapa}')

# Glavna funkcija
if __name__ == '__main__':
  #napravi()
  prikazi()
