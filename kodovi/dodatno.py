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
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.center_initializer \
     import kmeans_plusplus_initializer
from pyclustering.cluster.silhouette import silhouette

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt
from kmeans import kmeans_visualizer

# Parametri modela
datoteke = (*range(1, 5), 'x')
parametri = (('meanshift', '_tall'),
             ('meanshift', '_fall'),
             ('affinity', ''))
raspon = (*range(2, 6),)

# Prosecno pomeranje
def prospom(skup, mime):
  # Modelovanje bez autlajera
  klast = MeanShift(cluster_all=True, n_jobs=-1).fit(skup)
        
  # Cuvanje modela
  putanja = f'../modeli/meanshift/{mime}_meanshift_tall.joblib'
  os.makedirs(os.path.dirname(putanja), exist_ok=True)
  dump(klast, putanja)

  # Modelovanje sa autlajerima
  klast = MeanShift(cluster_all=False, n_jobs=-1).fit(skup)
        
  # Cuvanje modela
  putanja = f'../modeli/meanshift/{mime}_meanshift_fall.joblib'
  os.makedirs(os.path.dirname(putanja), exist_ok=True)
  dump(klast, putanja)

# Propagacija afiniteta
def propafin(skup, mime):
  # Modelovanje
  klast = AffinityPropagation().fit(skup)
        
  # Cuvanje modela
  putanja = f'../modeli/affinity/{mime}_affinity.joblib'
  os.makedirs(os.path.dirname(putanja), exist_ok=True)
  dump(klast, putanja)

# Fuzzy c-means
def fazi(skup, mime):
  # Za razlicite vrednosti c
  for c in raspon:
    # Inicijalizacija centara
    centri = kmeans_plusplus_initializer(skup, c).initialize()
    
    # Modelovanje
    klast = fcm(skup, centri).process()

    # Cuvanje modela
    putanja = f'../modeli/cmeans/{mime}_{c}fcm.joblib'
    os.makedirs(os.path.dirname(putanja), exist_ok=True)
    dump(klast, putanja)

# Algoritam G-sredina
def gmeansm(skup, mime):
  # Modelovanje
  klast = gmeans(skup).process()

  # Cuvanje modela
  putanja = f'../modeli/gmeans/{mime}_gmeans.joblib'
  os.makedirs(os.path.dirname(putanja), exist_ok=True)
  dump(klast, putanja)

# Pravljenje modela
def napravi():
  # Za svaki fajl iz projekta
  for i in datoteke:
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Pravljenje modela
    prospom(skup, f'GSM333056{i}')
    propafin(skup, f'GSM333056{i}')
    fazi(skup, f'GSM333056{i}')
    gmeansm(skup, f'GSM333056{i}')

# Prikaz modela
def prikazi():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for c in raspon:
      # Ucitavanje modela
      klast = load(f'../modeli/cmeans/GSM333056{i}_{c}fcm.joblib')
      klasteri = klast.get_clusters()

      # Racunanje senka koeficijenta
      skor = mean(silhouette(skup, klasteri).process().get_score())

      # Stampanje rezultata
      mapa = dict(enumerate(map(len, klasteri)))
      print(f'{i}-{c} ({skor:.2f}): {mapa}')

# Vizuelizacija modela
def vizuelizuj():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for c in raspon:
      # Ucitavanje modela
      klast = load(f'../modeli/cmeans/GSM333056{i}_{c}fcm.joblib')
      klasteri = klast.get_clusters()

      # Prikaz rezultata
      rez = kmeans_visualizer.show_clusters(skup, klasteri,
                                  None, display=False)
      putanja = f'../modeli/cslike/GSM333056{i}_{c}fcm'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      rez.savefig(putanja)
      plt.close(rez)

# Ocenjivanje modela
def oceni():
  # Citanje rezultata koji nisu nan
  with open('../pomocno/csilhouette.txt') as rez:
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
