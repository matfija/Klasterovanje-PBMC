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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt
from kmeans import kmeans_visualizer

# Parametri modela
datoteke = (*range(1, 5), 'x')
raspon = (*range(1, 6),)

# Gaussian mixture
def gaus(skup, mime):
  # Za razlicite vrednosti c
  for k in raspon:
    # Modelovanje; zbog velicine se ne cuva
    # model ovoga puta, vec samo oznake i mere
    klast = GaussianMixture(k).fit(skup)
    klast = klast.predict(skup), klast.bic(skup), klast.aic(skup)

    # Cuvanje modela
    putanja = f'../modeli/gauss/{mime}_{k}gauss.joblib'
    os.makedirs(os.path.dirname(putanja), exist_ok=True)
    dump(klast, putanja)

# Pravljenje modela
def napravi():
  # Za svaki fajl iz projekta
  for i in datoteke:
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Pravljenje modela
    gaus(skup, f'GSM333056{i}')

# Prikaz modela
def prikazi():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k in raspon:
      # Ucitavanje modela
      klast = load(f'../modeli/gauss/GSM333056{i}_{k}gauss.joblib')
      klasteri, bic, aic = klast

      # Racunanje senka koeficijenta
      skor = silhouette_score(skup, klasteri)\
             if max(klasteri) > 0 else 1.0

      # Pravljenje mape klastera
      mapa = {}
      for kl in klasteri:
        mapa[kl] = 1 if kl not in mapa else mapa[kl]+1

      # Stampanje rezultata
      print(f'{i}-{k}gauss ({skor:.2f}, {int(bic/1000000)}'
            f', {int(aic/1000000)}): {mapa}')

# Vizuelizacija modela
def vizuelizuj():
  # Za svaku kombinaciju parametara
  for i in datoteke:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    bics, aics = [], []
    for k in raspon:
      # Ucitavanje modela
      klast = load(f'../modeli/gauss/GSM333056{i}_{k}gauss.joblib')
      klasteri, bic, aic = klast
      bics.append(bic)
      aics.append(aic)

      # Mapiranje klastera u indekse
      klast = [[] for j in range(max(klasteri)+1)]
      for j in range(len(klasteri)):
        klast[klasteri[j]].append(j)
      klasteri = klast

      # Prikaz rezultata
      rez = kmeans_visualizer.show_clusters(skup, klasteri,
                                  None, display=False)
      putanja = f'../modeli/gausslike/GSM333056{i}_{k}gauss'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      rez.savefig(putanja)
      plt.close(rez)

    # Prikaz informacionih kriterijuma
    plt.plot(raspon, bics, label='BIC')
    plt.plot(raspon, aics, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('Broj klastera')
    putanja = f'../modeli/gausslike/GSM333056{i}_ic'
    os.makedirs(os.path.dirname(putanja), exist_ok=True)
    plt.savefig(putanja)
    plt.close()

# Ocenjivanje modela
def oceni():
  # Citanje rezultata koji nisu nan
  with open('../pomocno/gsilhouette.txt') as rez:
    rez = [*filter(lambda x: 'nan' not in x, rez.readlines())]

  # Izvlacenje informacija u trojkama
  def senka(rez):
    oz = rez.find('(')
    zz = rez.find(')')
    return (rez[:oz-1],
           *map(float, rez[oz+1:zz].split(', ')),
            eval(rez[zz+3:]))
  
  # Formatiranje rezultata
  rez = sorted(map(senka, rez), key=itemgetter(1), reverse=True)
  print('\n'.join(map(str, rez)))

# Glavna funkcija
if __name__ == '__main__':
  #napravi()
  #prikazi()
  #vizuelizuj()
  oceni()
