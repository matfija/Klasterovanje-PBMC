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
from numpy import mean, argmax
from sklearn.decomposition import NMF
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt
from kmeans import kmeans_visualizer

# Parametri modela
datoteke = (*range(1, 5), 'x')
raspon = (*range(2, 6),)
mere = ('rbf', 'nearest_neighbors')
strats = ('kmeans', 'discretize')

# Nenegativna faktorizacija matrice
def nmf(skup, mime):
  # Za razlicit broj klastera
  for k in raspon:
    # Modelovanje
    klast = NMF(k, max_iter=1000).fit(skup)

    # Cuvanje modela
    putanja = f'../modeli/spectral/nmf/{mime}_{k}nmf.joblib'
    os.makedirs(os.path.dirname(putanja), exist_ok=True)
    dump(klast, putanja)

# Spektralno klasterovanje
def spekt(skup, mime):
  # Za svaku kombinaciju parametara
  for k in raspon:
    for mera in mere:
      for strat in strats:
        # Modelovanje
        klast = SpectralClustering(n_clusters=k,
                                   affinity=mera,
                                   assign_labels=strat,
                                   n_jobs=-1).fit(skup)

        # Cuvanje modela
        poddir = f'{mera[0]}{strat[0]}'
        putanja = f'../modeli/spectral/{poddir}/'\
                  f'{mime}_{k}{poddir}.joblib'
        os.makedirs(os.path.dirname(putanja), exist_ok=True)
        dump(klast, putanja)

# Spektralno nad NMF
def spektnmf(mime):
  # Za svaku kombinaciju parametara
  for k in raspon:
    skup = load(f'../modeli/spectral/nmf/{mime}_{k}nmf.joblib')
    skup = skup.components_.T
    for mera in mere:
      for strat in strats:
        # Modelovanje
        klast = SpectralClustering(n_clusters=k,
                                   affinity=mera,
                                   assign_labels=strat,
                                   n_jobs=-1).fit(skup)

        # Cuvanje modela
        poddir = f'{mera[0]}{strat[0]}'
        putanja = f'../modeli/spectral/nmf{poddir}/'\
                  f'{mime}_{k}nmf{poddir}.joblib'
        os.makedirs(os.path.dirname(putanja), exist_ok=True)
        dump(klast, putanja)

# Pravljenje modela
def napravi():
  # Za svaki fajl iz projekta
  for i in datoteke:
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Pravljenje modela
    nmf(skup.T, f'GSM333056{i}')
    spekt(skup, f'GSM333056{i}')
    spektnmf(f'GSM333056{i}')

# Prikaz NMF modela
def prikazin():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k in raspon:
      # Ucitavanje modela
      klast = load(f'../modeli/spectral/nmf/'
                   f'GSM333056{i}_{k}nmf.joblib')
      klasteri = argmax(klast.components_, axis=0)

      # Racunanje senka koeficijenta
      skor = silhouette_score(skup, klasteri)

      # Pravljenje mape klastera
      mapa = {}
      for kl in klasteri:
        mapa[kl] = 1 if kl not in mapa else mapa[kl]+1

      # Stampanje rezultata
      print(f'{i}-{k}nmf ({skor:.2f}): {mapa}')

# Vizuelizacija NMF modela
def vizuelizujn():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for k in raspon:
      # Ucitavanje modela
      klast = load(f'../modeli/spectral/nmf/'
                   f'GSM333056{i}_{k}nmf.joblib')
      klasteri = argmax(klast.components_, axis=0)

      # Mapiranje klastera u indekse
      klast = [[] for j in range(max(klasteri)+1)]
      for j in range(len(klasteri)):
        klast[klasteri[j]].append(j)
      klasteri = klast

      # Prikaz rezultata
      rez = kmeans_visualizer.show_clusters(skup, klasteri,
                                  None, display=False)
      putanja = f'../modeli/spectslike/GSM333056{i}_{k}nmf'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      rez.savefig(putanja)
      plt.close(rez)

# Prikaz spektralnog modela
def prikazis(nmf=''):
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k in raspon:
      for mera in mere:
        for strat in strats:
          # Ucitavanje modela
          poddir = f'{mera[0]}{strat[0]}'
          klast = load(f'../modeli/spectral/{nmf}{poddir}/'
                       f'GSM333056{i}_{k}{nmf}{poddir}.joblib')
          klasteri = klast.labels_

          # Racunanje senka koeficijenta
          skor = silhouette_score(skup, klasteri)

          # Pravljenje mape klastera
          mapa = {}
          for kl in klasteri:
            mapa[kl] = 1 if kl not in mapa else mapa[kl]+1

          # Stampanje rezultata
          print(f'{i}-{k}{nmf}{poddir} ({skor:.2f}): {mapa}')

# Vizuelizacija spektralnog modela
def vizuelizujs(nmf=''):
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for k in raspon:
      for mera in mere:
        for strat in strats:
          # Ucitavanje modela
          poddir = f'{mera[0]}{strat[0]}'
          klast = load(f'../modeli/spectral/{nmf}{poddir}/'
                       f'GSM333056{i}_{k}{nmf}{poddir}.joblib')
          klasteri = klast.labels_

          # Mapiranje klastera u indekse
          klast = [[] for j in range(max(klasteri)+1)]
          for j in range(len(klasteri)):
            klast[klasteri[j]].append(j)
          klasteri = klast

          # Prikaz rezultata
          rez = kmeans_visualizer.show_clusters(skup, klasteri,
                                      None, display=False)
          putanja = f'../modeli/spectslike/GSM333056{i}_{k}{nmf}{poddir}'
          os.makedirs(os.path.dirname(putanja), exist_ok=True)
          rez.savefig(putanja)
          plt.close(rez)

# Ocenjivanje modela
def oceni():
  # Citanje rezultata koji nisu nan
  with open('../pomocno/ssilhouette.txt') as rez:
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
  #vizuelizujn()
  #vizuelizujs()
  #vizuelizujs('nmf')
  #prikazin()
  #prikazis()
  #prikazis('nmf')
  oceni()
