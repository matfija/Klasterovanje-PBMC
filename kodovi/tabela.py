# Datoteka sa senka koeficijentima
with open('../pomocno/svesenke.txt', 'r') as dat:
  senke = dat.readlines()

# Izvlacenje informacija u petorkama
# 1-2-mean-man (0.47): {0: 501, 1: 17}
# fajl-klast-metod-mera (koef): mapa
def senka(linija):
  # Nalazenje zagrada
  oz = linija.find('(')
  zz = linija.find(')')

  # Prvi deo linije
  fajl, klast, metod, mera \
        = linija[:oz-1].split('-')

  # Senka koeficijent
  senka = float(linija[oz+1:zz])

  # Vracanje rezultata
  return fajl, klast, metod, mera, senka

# Pretvaranje linija u petorke
senke = [*map(senka, senke)]

# Pretvaranje petorki u mapu cetvorki
mapa = {'1': [], '2': [], '3': [], '4': [], 'x': []}
for fajl, klast, metod, mera, senka in senke:
  mapa[fajl].append((klast, metod, mera, senka))

# Datoteka sa rezultujucom tabelom
with open('../pomocno/tabsenke.csv', 'w') as dat:
  # Upisivanje zaglavlja
  dat.write('Klast,Metod,Mera,GSM1,GSM2,GSM3,GSM4,GSMx\n')

  # Upisivanje svakog modela
  for d1, d2, d3, d4, dx in zip(*(mapa[m] for m in mapa)):
    klast, metod, mera, s1 = d1
    _    , _    , _   , s2 = d2
    _    , _    , _   , s3 = d3
    _    , _    , _   , s4 = d4
    _    , _    , _   , sx = dx
    dat.write(f'{klast},{metod},{mera},{s1},{s2},{s3},{s4},{sx}\n')
