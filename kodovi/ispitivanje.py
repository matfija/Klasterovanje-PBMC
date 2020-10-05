# Biblioteka za citanje
import pandas as pd

# Biblioteka za rad sa OS
import os

# Biblioteka za operatore
from operator import itemgetter

# Biblioteka za cuvanje modela
import dill
from joblib import dump, load

# Biblioteka za dimenzionu redukciju
from sklearn.manifold import TSNE

# Biblioteka za klasterovanje
from sklearn.metrics import silhouette_score

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt
from kmeans import kmeans_visualizer

# Pakovanje t-SNE skupova
def zapakuj():
  # Za svaki fajl iz projekta
  spisak = ['Pre', 'Disc_Early', 'Disc_Resp', 'Disc_AR']
  for i, desc in enumerate(spisak, start=3330561):
    # Ucitavanje skupa
    skup = pd.read_csv(f'../GSM{i}/GSM{i}_PBMC_{desc}_t.csv',
                       index_col=0).values

    # Pakovanje skupa
    putanja = f'../modeli/tsne/GSM{i}.joblib'
    os.makedirs(os.path.dirname(putanja), exist_ok=True)
    dump(skup, putanja)
    
    # Transformacija skupa
    skup = TSNE(n_jobs=-1).fit_transform(skup)

    # Pakovanje skupa
    putanja = f'../modeli/tsne/GSM{i}t.joblib'
    os.makedirs(os.path.dirname(putanja), exist_ok=True)
    dump(skup, putanja)

  # Ucitavanje spojenog skupa
  skup = pd.read_csv(f'../GSM333056x/GSM333056x.csv',
                     index_col=0).values

  # Pakovanje skupa
  putanja = f'../modeli/tsne/GSM333056x.joblib'
  os.makedirs(os.path.dirname(putanja), exist_ok=True)
  dump(skup, putanja)

  # Transformacija skupa
  skup = TSNE(n_jobs=-1).fit_transform(skup)

  # Pakovanje skupa
  putanja = f'../modeli/tsne/GSM333056xt.joblib'
  os.makedirs(os.path.dirname(putanja), exist_ok=True)
  dump(skup, putanja)

# Ispitivanje kvaliteta podele po datotekama
def podela():
  # Ucitavanje spojenog skupa
  skup = pd.read_csv(f'../GSM333056x/GSM333056x.csv',
                     index_col=0)

  # Oznacavanje klastera
  klasteri = [*map(int, map(itemgetter(9), skup.index))]

  # Uproscavanje skupa
  skup = skup.values

  # Racunanje senka koeficijenta za svaku meru
  for mera in ('manhattan', 'euclidean', 'cosine'):
    skor = silhouette_score(skup, klasteri, metric=mera)
    print(f'GSM333056x-{mera[:3]}: {skor:.2f}')

  # Ucitavanje t-SNE redukcije
  skup = load(f'../modeli/tsne/GSM333056xt.joblib')

  # Mapiranje klastera u indekse
  klast = [[] for i in range(max(klasteri)+1)]
  for i in range(len(klasteri)):
    klast[klasteri[i]].append(i)
  klasteri = klast

  # Prikaz rezultata
  rez = kmeans_visualizer.show_clusters(skup, klasteri,
                              None, display=False)
  rez.savefig(f'../pomocno/GSM333056x')
  plt.close(rez)

# Glavna funkcija
if __name__ == '__main__':
  #zapakuj()
  podela()
