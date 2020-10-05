# Biblioteka za rad sa OS
import os

# Bibliote za matematiku
import numpy as np

# Biblioteka za slucajnost
from numpy.random import seed
seed(0)

# Biblioteka za cuvanje modela
import dill
from joblib import dump, load

# Biblioteka za operatore
from operator import itemgetter

# Biblioteka za klasterovanje
from minisom import MiniSom
from sklearn.metrics import silhouette_score

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt
from kmeans import kmeans_visualizer

# Skup parametara
datoteke = (*range(1, 5), 'x')
raspon = (*range(2, 6),)
mere = ('manhattan', 'euclidean', 'cosine')

# Samoorganizujuca mapa
def som(skup, mime):
  # Za sve kombinacije broja klastera,
  # povezanosti i mera rastojanja
  for k in raspon:
    for mera in mere:
      # Modelovanje
      klast = MiniSom(1, k, skup.shape[1],
                      sigma=.5, learning_rate=.5,
                      activation_distance=mera)
      klast.train(skup, 10*skup.shape[0])
        
      # Cuvanje modela
      putanja = f'../modeli/som/{mime}_{k}_{mera[:3]}.joblib'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      dump(klast, putanja)

# Pravljenje modela
def napravi():
  # Za svaki fajl iz projekta
  for i in datoteke:
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Pravljenje modela
    som(skup, f'GSM333056{i}')

# Prikaz modela
def prikazi():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k in raspon:
      for mera in mere:
        # Ucitavanje modela
        klast = load(f'../modeli/som/GSM333056{i}'
                     f'_{k}_{mera[:3]}.joblib')

        # Izvlacenje klastera
        pobednici = np.array([klast.winner(x) for x in skup]).T
        klasteri = np.ravel_multi_index(pobednici, (1, k))
      
        # Racunanje senka koeficijenta
        skor = silhouette_score(skup, klasteri, metric=mera)

        # Pravljenje mape klastera
        mapa = {}
        for kl in klasteri:
          mapa[kl] = 1 if kl not in mapa else mapa[kl]+1

        # Stampanje rezultata
        print(f'{i}-{k}-{mera[:3]} ({skor:.2f}): {mapa}')

# Vizuelizacija modela
def vizuelizuj():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    tsne = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for k in raspon:
      for mera in mere:
        # Ucitavanje modela
        klast = load(f'../modeli/som/GSM333056{i}'
                     f'_{k}_{mera[:3]}.joblib')

        # Izvlacenje klastera
        pobednici = np.array([klast.winner(x) for x in skup]).T
        klasteri = np.ravel_multi_index(pobednici, (1, k))

        # Mapiranje klastera u indekse
        klast = [[] for j in range(max(klasteri)+1)]
        for j in range(len(klasteri)):
          klast[klasteri[j]].append(j)
        klasteri = klast

        # Prikaz rezultata
        rez = kmeans_visualizer.show_clusters(tsne, klasteri,
                                    None, display=False)
        putanja = f'../modeli/somslike/GSM333056{i}_{k}_{mera[:3]}'
        os.makedirs(os.path.dirname(putanja), exist_ok=True)
        rez.savefig(putanja)
        plt.close(rez)

# Ocenjivanje modela
def oceni():
  # Citanje rezultata koji nisu nan
  with open('../pomocno/somsilhouette.txt') as rez:
    rez = [*filter(lambda x: 'nan' not in x, rez.readlines())]

  # Izvlacenje informacija u trojkama
  def senka(rez):
    oz = rez.find('(')
    zz = rez.find(')')
    return rez[:oz-1], float(rez[oz+1:zz]), eval(rez[zz+3:])
  
  # Formatiranje rezultata
  rez = sorted(map(senka, rez), key=itemgetter(1), reverse=True)
  print('\n'.join(map(str, rez)))

# Glavna funkcija
if __name__ == '__main__':
  #napravi()
  #prikazi()
  #vizuelizuj()
  oceni()
