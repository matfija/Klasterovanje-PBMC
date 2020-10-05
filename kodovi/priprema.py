# Ukljucivanje modula za JSON
import json

# Ucitavanje cestih ljudskih gena
with open('../pomocno/hg38_ensg.json', 'r') as dat:
  geni = json.load(dat)

# Inicijalizacija liste odbacenih
odbaceni = []

# Za svaki fajl iz projekta
spisak = ['Pre', 'Disc_Early', 'Disc_Resp', 'Disc_AR']
for i, desc in enumerate(spisak, start=3330561):
  # Ucitavanje podataka i ciljne datoteke
  with open(f'../GSM{i}/GSM{i}_PBMC_{desc}.csv', 'r') as stari,\
       open(f'../GSM{i}/GSM{i}_PBMC_{desc}_p.csv', 'w') as novi:

    # Pocetni indeks
    j = 0

    # Zamena naziva kolone
    def zameni(col):
      global j
      j += 1
      return f'GSM{i}_{j}'

    # Popunjavanje prvog reda atributima (celijama)
    prva = stari.readline().split(',')
    prva = [prva[0]] + [*map(zameni, prva[1:])]
    prva = ','.join(prva) + '\n'
    novi.write(prva)

    # Popunjavanje ostalih redova genima
    for linija in stari:
      # Izdvajanje naziva gena
      pozgen = linija.find(',')
      gen = linija[:pozgen]

      # Odbacivanje gena koji nisu cesti
      if gen not in geni:
        odbaceni.append(gen)
        continue

      # Izdvajanje odgovarajuceg ENSG-a
      ensg = geni[gen]
      
      # Upisivanje rezultata u novu datoteku
      novi.write(ensg + linija[pozgen:])

# Izvestavanje o odbacenim genima
with open('../pomocno/odbaceni.txt', 'w') as izvestaj:
  izvestaj.write('\n'.join(odbaceni))
