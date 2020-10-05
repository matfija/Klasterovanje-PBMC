# Biblioteka za rad sa OS
import os

# Biblioteka za slucajnost
from random import sample
from numpy.random import seed
seed(0)

# Biblioteka za cuvanje modela
import dill
from joblib import dump, load

# Biblioteka za operatore
from operator import itemgetter

# Biblioteka za klasterovanje
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
kmeans_visualizer._kmeans_visualizer__draw_centers = lambda *args: None
kmeans_visualizer._kmeans_visualizer__draw_rays = lambda *args: None
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer \
     import kmeans_plusplus_initializer
from pyclustering.cluster.silhouette import silhouette

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt

# Biblioteka za mere
from pyclustering.utils.metric import type_metric
from pyclustering.utils.metric import distance_metric

# Kosinusno rastojanje
from numpy import inner, mean
from numpy.linalg import norm
def cosine(a, b):
  return 1-inner(a, b)/(norm(a) * norm(b))

# Razlicite mere rastojanja
mere = ((distance_metric(type_metric.MANHATTAN), 'man'),
        (distance_metric(type_metric.EUCLIDEAN), 'euc'),
        (distance_metric(type_metric.USER_DEFINED,
                         func=cosine), 'cos'))

# Parametri modela
datoteke = (*range(1, 5), 'x')
raspon = (*range(2, 6),)

# K-means model
def kmeansm(skup, mime):
  # Za razlicite vrednosti k
  for k in raspon:
    # Inicijalizacija centara
    centri = kmeans_plusplus_initializer(skup, k).initialize()
    
    # Za razlicite mere rastojanja
    for mera, merime in mere:
      # Modelovanje
      klast = kmeans(skup, centri, metric=mera).process()
      
      # Cuvanje modela
      putanja = f'../modeli/repcent/{k}means/'\
                f'{mime}_{k}means_{merime}.joblib'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      dump(klast, putanja)

# K-median model
def kmediansm(skup, mime):
  # Za razlicite vrednosti k
  for k in raspon:
    # Inicijalizacija centara
    centri = kmeans_plusplus_initializer(skup, k).initialize()
    
    # Za razlicite mere rastojanja
    for mera, merime in mere:
      # Modelovanje
      klast = kmedians(skup, centri, metric=mera).process()
      
      # Cuvanje modela
      putanja = f'../modeli/repcent/{k}medians/'\
                f'{mime}_{k}medians_{merime}.joblib'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      dump(klast, putanja)

# K-medoid model
def kmedoidsm(skup, mime):
  # Za razlicite vrednosti k
  for k in raspon:
    # Inicijalizacija centara
    centri = sample(range(skup.shape[0]), k)
    
    # Za razlicite mere rastojanja
    for mera, merime in mere:
      # Modelovanje
      klast = kmedoids(skup, centri, metric=mera).process()
      
      # Cuvanje modela
      putanja = f'../modeli/repcent/{k}medoids/'\
                f'{mime}_{k}medoids_{merime}.joblib'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      dump(klast, putanja)

# Pravljenje modela
def napravi():
  # Za svaki fajl iz projekta
  for i in datoteke:
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Pravljenje modela
    kmeansm(skup, f'GSM333056{i}')
    kmediansm(skup, f'GSM333056{i}')
    kmedoidsm(skup, f'GSM333056{i}')

# Prikaz modela
def prikazi():
  # Za svaku kombinaciju algoritma,
  # skupa, parametra k i metrike
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k in raspon:
      for alg in ('means', 'medians', 'medoids'):
        for mera, merime in mere:
          # Ucitavanje modela
          klast = load(f'../modeli/repcent/{k}{alg}/GSM333056'
                       f'{i}_{k}{alg}_{merime}.joblib')
          klasteri = klast.get_clusters()

          # Racunanje senka koeficijenta
          skor = mean(silhouette(skup, klasteri,
                        metric=mera).process().get_score())

          # Stampanje rezultata
          mapa = dict(enumerate(map(len, klasteri)))
          print(f'{i}-{k}-{alg[:4]}-{merime} ({skor:.2f}): {mapa}')

# Vizuelizacija modela
def vizuelizuj():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for k in raspon:
      for alg in ('means', 'medians', 'medoids'):
        for mera in ('man', 'euc', 'cos'):
          # Ucitavanje modela
          klast = load(f'../modeli/repcent/{k}{alg}/GSM333056'
                       f'{i}_{k}{alg}_{mera}.joblib')
          klasteri = klast.get_clusters()

          # Prikaz rezultata
          rez = kmeans_visualizer.show_clusters(skup, klasteri,
                                      None, display=False)
          putanja = f'../modeli/kslike/GSM333056{i}_{k}{alg}_{mera}'
          os.makedirs(os.path.dirname(putanja), exist_ok=True)
          rez.savefig(putanja)
          plt.close(rez)

# Ocenjivanje modela
def oceni():
  # Citanje rezultata koji nisu nan
  with open('../pomocno/ksilhouette.txt') as rez:
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
