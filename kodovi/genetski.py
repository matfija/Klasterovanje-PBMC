# Biblioteka za rad sa OS
import os

# Biblioteka za slucajnost
from numpy.random import seed
seed(0)

# Biblioteka za cuvanje modela
import dill
from joblib import dump, load

# Biblioteka za operatore
from operator import itemgetter

# Biblioteka za klasterovanje
from numpy import mean
from pyclustering.cluster.ga import genetic_algorithm,\
                                    ga_observer,\
                                    ga_visualizer
from pyclustering.cluster.silhouette import silhouette

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt

# Parametri modela
datoteke = (*range(1, 5), 'x')
raspon = (*range(2, 6),)

# Genetski algoritam
def genetski(skup, mime):
  # Za razlicit broj klastera
  for k in raspon:
    # Modelovanje
    klast = genetic_algorithm(data=skup,
                              count_clusters=k,
                              chromosome_count=20,
                              population_count=100,
                              select_coeff=0.0001,
                              observer=ga_observer(True,
                                                   True,
                                                   True))
    klast.process()

    # Cuvanje modela
    putanja = f'../modeli/genetic/{mime}_{k}ga.joblib'
    os.makedirs(os.path.dirname(putanja), exist_ok=True)
    dump(klast, putanja)

# Pravljenje modela
def napravi():
  # Za svaki fajl iz projekta
  for i in datoteke:
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Pravljenje modela
    genetski(skup, f'GSM333056{i}')

# Prikaz modela
def prikazi():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k in raspon:
      # Ucitavanje modela
      klast = load(f'../modeli/genetic/GSM333056{i}_{k}ga.joblib')
      klasteri = klast.get_clusters()

      # Racunanje senka koeficijenta
      skor = mean(silhouette(skup, klasteri).process().get_score())

      # Stampanje rezultata
      mapa = dict(enumerate(map(len, klasteri)))
      print(f'{i}-{k} ({skor:.2f}): {mapa}')

# Vizuelizacija modela
def vizuelizuj():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for k in raspon:
      # Ucitavanje modela
      klast = load(f'../modeli/genetic/GSM333056{i}_{k}ga.joblib')

      # Prikaz rezultata
      ga_visualizer.show_clusters(skup, klast.get_observer())

# Glavna funkcija
if __name__ == '__main__':
  #napravi()
  #vizuelizuj()
  prikazi()
