# Biblioteka za rad sa fajl sistemom
import os

# Biblioteka za cuvanje modela
from joblib import load

# Biblioteka za matematiku
from numpy import count_nonzero

# Biblioteka za iteraciju
from itertools import product

# Biblioteka za operatore
from operator import itemgetter

# Biblioteka za klasterovanje
from sklearn.metrics import silhouette_score

# Biblioteka za vizuelizaciju
import matplotlib.pyplot as plt
from kmeans import kmeans_visualizer

# Parametri modela
dats = (*range(1, 5), 'x')
raspon = (*range(2, 6),)
mere = ('cos', 'corr')
kritb = ('i2', 'h2', 'e1', 'g1')
kritp = ('i2', 'h2', 'wslink', 'wclink')
mods = ('GSM3330561_2rbr_cos_h2.sol',
        'GSM3330561_3rbr_cos_h2.sol',
        'GSM3330561_4rbr_cos_h2.sol',
        'GSM3330561_5rbr_cos_h2.sol')

# Priprema datoteke
def csv2mat(skup, ime):
  # Upisivanje podataka u fajl
  with open(f'../modeli/tsne/{ime}.mat', 'w') as dat:
    # Prvi red sa dimenzijama i brojem nenula
    print(*skup.shape, count_nonzero(skup), file=dat)

    # Ostali redovi parovi indeksa i vrednosti
    skup = '\n'.join(map(lambda x: '\t'. join(map(
                         lambda x: f'{x[0]} {x[1]}',
                         filter(lambda x: x[1] > 0,
                                enumerate(x, 1)))), skup))
    dat.write(skup)

# Priprema datoteka
def pripremi():
  # Za svaki fajl iz projekta
  for i in dats:
    # Ucitavanje skupa
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')

    # Priprema datoteke
    csv2mat(skup, f'GSM333056{i}')

# Bisekcija k-sredina
def bisekcija():
  # Priprema direktorijuma
  os.makedirs(os.path.dirname('../modeli/rbr/'), exist_ok=True)

  # Modelovanje za sve kombinacije parametara
  for i, k, m, c in product(dats, raspon, mere, kritb):
    if m == 'corr' and c != 'e1': continue
    os.system(f'vcluster -clmethod=rbr -sim={m} -crfun={c} -seed=0 '
              f'-clustfile=../modeli/rbr/GSM333056{i}_{k}rbr_{m}'
              f'_{c}.sol ../modeli/tsne/GSM333056{i}.mat {k}')

# Prikaz modela
def prikazib():
  # Za sve kombinacije parametara
  for i in dats:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k, m, c in product(raspon, mere, kritb):
      if m == 'corr' and c != 'e1': continue
      
      # Citanje oznaka iz datoteke
      with open(f'../modeli/rbr/GSM333056{i}'
                f'_{k}rbr_{m}_{c}.sol', 'r') as dat:
        klasteri = [*map(int, dat.readlines())]

      # Racunanje senka koeficijenta
      skor = silhouette_score(skup, klasteri, metric='cosine')

      # Pravljenje mape klastera
      mapa = {}
      for kl in klasteri:
        mapa[kl] = 1 if kl not in mapa else mapa[kl]+1

      # Stampanje rezultata
      print(f'{i}-{k}rbr-{m}-{c} ({skor:.2f}): {mapa}')

# Vizuelizacija modela
def vizuelizujb():
  # Za sve kombinacije parametara
  for i in dats:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for k, m, c in product(raspon, mere, kritb):
      if m == 'corr' and c != 'e1': continue
      
      # Citanje oznaka iz datoteke
      with open(f'../modeli/rbr/GSM333056{i}'
                f'_{k}rbr_{m}_{c}.sol', 'r') as dat:
        klasteri = [*map(int, dat.readlines())]

      # Mapiranje klastera u indekse
      klast = [[] for j in range(max(klasteri)+1)]
      for j in range(len(klasteri)):
        klast[klasteri[j]].append(j)
      klasteri = klast

      # Prikaz rezultata
      rez = kmeans_visualizer.show_clusters(skup, klasteri,
                                  None, display=False)
      putanja = f'../modeli/rbrslike/GSM333056{i}_{k}rbr_{m}_{c}'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      rez.savefig(putanja)
      plt.close(rez)

# Pristrasno sakupljajuce
def pristsak():
  # Priprema direktorijuma
  os.makedirs(os.path.dirname('../modeli/prist/'), exist_ok=True)

  # Modelovanje za sve kombinacije parametara
  for i, k, m, c in product(dats, raspon, mere, kritp):
    os.system(f'vcluster -clmethod=bagglo -sim={m} -crfun={c} -seed=0 '
              f'-clustfile=../modeli/prist/GSM333056{i}_{k}prist_{m}'
              f'_{c}.sol ../modeli/tsne/GSM333056{i}.mat {k}')

# Prikaz modela
def prikazip():
  # Za sve kombinacije parametara
  for i in dats:
    skup = load(f'../modeli/tsne/GSM333056{i}.joblib')
    for k, m, c in product(raspon, mere, kritp):
      # Citanje oznaka iz datoteke
      with open(f'../modeli/prist/GSM333056{i}'
                f'_{k}prist_{m}_{c}.sol', 'r') as dat:
        klasteri = [*map(int, dat.readlines())]

      # Racunanje senka koeficijenta
      skor = silhouette_score(skup, klasteri, metric='cosine')

      # Pravljenje mape klastera
      mapa = {}
      for kl in klasteri:
        mapa[kl] = 1 if kl not in mapa else mapa[kl]+1

      # Stampanje rezultata
      print(f'{i}-{k}prist-{m}-{c} ({skor:.2f}): {mapa}')
  
# Vizuelizacija modela
def vizuelizujp():
  # Za sve kombinacije parametara
  for i in dats:
    skup = load(f'../modeli/tsne/GSM333056{i}t.joblib')
    for k, m, c in product(raspon, mere, kritp):
      # Citanje oznaka iz datoteke
      with open(f'../modeli/prist/GSM333056{i}'
                f'_{k}prist_{m}_{c}.sol', 'r') as dat:
        klasteri = [*map(int, dat.readlines())]

      # Mapiranje klastera u indekse
      klast = [[] for j in range(max(klasteri)+1)]
      for j in range(len(klasteri)):
        klast[klasteri[j]].append(j)
      klasteri = klast

      # Prikaz rezultata
      rez = kmeans_visualizer.show_clusters(skup, klasteri,
                                  None, display=False)
      putanja = f'../modeli/pristslike/GSM333056{i}_{k}prist_{m}_{c}'
      os.makedirs(os.path.dirname(putanja), exist_ok=True)
      rez.savefig(putanja)
      plt.close(rez)

# Prikaz gCLUTO
def prikazic():
  # Za sve kombinacije parametara
  skup = load(f'../modeli/tsne/GSM3330561.joblib')
  tsne = load(f'../modeli/tsne/GSM3330561t.joblib')
  for mod in mods:
    # Citanje oznaka iz datoteke
    with open(f'../modeli/gcluto/{mod}', 'r') as dat:
        klasteri = [*map(int, dat.readlines())]

    # Racunanje senka koeficijenta
    skor = silhouette_score(skup, klasteri, metric='cosine')

    # Mapiranje klastera u indekse
    klast = [[] for j in range(max(klasteri)+1)]
    for j in range(len(klasteri)):
      klast[klasteri[j]].append(j)
    klasteri = klast

    # Prikaz rezultata
    rez = kmeans_visualizer.show_clusters(tsne, klasteri,
                                None, display=False)
    rez.savefig(f'../modeli/gcluto/{mod[:-3]}')
    plt.close(rez)

    # Stampanje rezultata
    mapa = dict(enumerate(map(len, klasteri)))
    print(f'{mod[9:]} ({skor:.2f}): {mapa}')

# Glavna funkcija
if __name__ == '__main__':
  #pripremi()
  #bisekcija()
  #prikazib()
  #vizuelizujb()
  #pristsak()
  #prikazip()
  #vizuelizujp()
  prikazic()
