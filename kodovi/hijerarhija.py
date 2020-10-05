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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt
from kmeans import kmeans_visualizer

# Parametri modela
datoteke = (*range(1, 5), 'x')
raspon = (*range(2, 6),)
mere = ('manhattan', 'euclidean', 'cosine')
veze = ('ward', 'complete', 'average', 'single')

# Sakupljajuci hijerarhijski model
def sakupljajuce(skup, mime):
  # Za sve kombinacije broja klastera,
  # povezanosti i mera rastojanja
  for k in raspon:
    for veza in veze: 
      for mera in mere:
        # Ward moze samo euklidsko
        if veza == 'ward' and mera != 'euclidean':
          continue
        
        # Modelovanje
        klast = AgglomerativeClustering(n_clusters=k,
                                        affinity=mera,
                                        linkage=veza).fit(skup)
        
        # Cuvanje modela
        putanja = f'../modeli/agglo/{k}{veza}/{mime}_'\
                  f'{k}{veza}_{mera[:3]}.joblib'
        os.makedirs(os.path.dirname(putanja), exist_ok=True)
        dump(klast, putanja)

# Pravljenje modela
def napravi():
  # Za svaki fajl iz projekta
  for i in datoteke:
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Pravljenje modela
    sakupljajuce(skup, f'GSM333056{i}')

# Prikaz modela
def prikazi():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k in raspon:
      for veza in veze: 
        for mera in mere:
          # Ward moze samo euklidsko
          if veza == 'ward' and mera != 'euclidean':
            continue
          
          # Ucitavanje modela
          klast = load(f'../modeli/agglo/{k}{veza}/GSM333056'
                       f'{i}_{k}{veza}_{mera[:3]}.joblib')
          klasteri = klast.labels_
      
          # Racunanje senka koeficijenta
          skor = silhouette_score(skup, klasteri, metric=mera)

          # Pravljenje mape klastera
          mapa = {}
          for kl in klasteri:
            mapa[kl] = 1 if kl not in mapa else mapa[kl]+1

          # Stampanje rezultata
          print(f'{i}-{k}-{veza[:4]}-{mera[:3]} ({skor:.2f}): {mapa}')

# Vizuelizacija modela
def vizuelizuj():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for k in raspon:
      for veza in veze: 
        for mera in mere:
          # Ward moze samo euklidsko
          if veza == 'ward' and mera != 'euclidean':
            continue
          
          # Ucitavanje modela
          klast = load(f'../modeli/agglo/{k}{veza}/GSM333056'
                       f'{i}_{k}{veza}_{mera[:3]}.joblib')
          klasteri = klast.labels_

          # Mapiranje klastera u indekse
          klast = [[] for j in range(max(klasteri)+1)]
          for j in range(len(klasteri)):
            klast[klasteri[j]].append(j)
          klasteri = klast

          # Prikaz rezultata
          rez = kmeans_visualizer.show_clusters(skup, klasteri,
                                      None, display=False)
          putanja = '../modeli/aggloslike/GSM333056'\
                    f'{i}_{k}{veza}_{mera[:3]}'
          os.makedirs(os.path.dirname(putanja), exist_ok=True)
          rez.savefig(putanja)
          plt.close(rez)

# Ocenjivanje modela
def oceni():
  # Citanje rezultata koji nisu nan
  with open('../pomocno/agglosilhouette.txt') as rez:
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
