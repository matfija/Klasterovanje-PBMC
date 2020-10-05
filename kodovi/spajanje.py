# Biblioteka za citanje
import pandas as pd

# Udeo nula u kolonama kao parametar
p = 0.01

# Inicijalizacija liste skupova
skupovi = []

# Stampanje metapodataka
print('Samo cesti geni:')

# Za svaki fajl iz projekta
spisak = ['Pre', 'Disc_Early', 'Disc_Resp', 'Disc_AR']
for i, desc in enumerate(spisak, start=3330561):
  # Popravljeni indeks
  j = i - 3330560
  
  # Ucitavanje skupa
  skup = pd.read_csv(f'../GSM{i}/GSM{i}_PBMC_{desc}_p.csv',
                     index_col=0)
  
  # Stampanje metapodataka
  print(f'{j}. {desc} -> {skup.shape}')

  # Transponovanje datafrejma
  skup = skup.transpose()

  # Uklanjanje celija koje nemaju bar 1000 transkripata
  # ili nemaju bar 500 ispoljenih (nenula) gena
  uslov1 = skup.sum(axis=1) >= 1000
  uslov2 = skup.apply(lambda x: x > 0).sum(axis=1) >= 500
  skup = skup[uslov1 & uslov2]

  # Cuvanje skupa u listi
  skupovi.append(skup)

# Stampanje metapodataka
print()

# Spajanje tabela nadovezivanjem
skupovis = pd.concat(skupovi)

# Uklanjanje gena ispoljenih u manje od p% celija
preko = p*skupovis.shape[0]
uslov = skupovis.apply(lambda x: x > 0).sum(axis=0).apply(lambda x: x > preko)
nule = '\n'.join(skupovis.columns[~uslov])
skupovis = skupovis[skupovis.columns[uslov]]

# Izvestavanje o nula genima
with open('../GSM333056x/GSM333056x_nule.txt', 'w') as izvestaj:
  izvestaj.write(nule)

# Izvestavanje o nenula genima
nenule = '\n'.join(skupovis.columns)
with open('../GSM333056x/GSM333056x_nenule.txt', 'w') as izvestaj:
  izvestaj.write(nenule)

# Stampanje metapodataka
print('Spojena datoteka')
print(f'GSM333056x -> {skupovis.shape}')
print()

# Cuvanje rezultata spajanja
skupovis.to_csv('../GSM333056x/GSM333056x.csv')

# Inicijalizacija ispoljenost
ispoljenost = pd.DataFrame(dtype='int')

# Stampanje metapodataka
print('Sa dovoljno ispoljavanja:')

# Za svaki fajl iz projekta
for i, desc in enumerate(spisak, start=3330561):
  # Popravljeni indeks
  j = i - 3330561
  
  # Tekuci rezultat ispoljenosti
  ispoljenost[f'GSM{i}'] = skupovi[j].sum(axis=0)
  
  # Izdvajanje zajednickih atributa
  skupovi[j] = skupovi[j][skupovis.columns]
  
  # Stampanje metapodataka
  print(f'{j+1}. {desc} -> {skupovi[j].shape}')

  # Cuvanje rezultata transponovanja
  skupovi[j].to_csv(f'../GSM{i}/GSM{i}_PBMC_{desc}_t.csv')

# Cuvanje rezultata ispoljenosti
ispoljenost.to_csv('../GSM333056x/GSM333056x_ispoljenost.csv')
