# Ukljucivanje modula za JSON
import json

# Ukljucivanje modula za regularne izraze
import re

# Ucitavanje cestih ljudskih gena
with open('../pomocno/common_human_list.csv', 'r') as gen:
  geni = gen.readlines()[1:]

# Inijalizacija liste dupliranih
dupli = []

# Popunjavanje mape
mapa = {}
for gen in geni:
  # Podela reda po zapeti
  gen = gen.split(',')

  # Obrada ENSG-a
  def fensg(ensg):
    poz = re.search('[1-9]', ensg).start()
    return 'E' + ensg[poz:]

  # Izdvajanje podataka
  ensg = fensg(gen[0])
  hg38 = gen[3][7:]
  ensemble = gen[4]

  # Upisivanje u mapu
  if hg38 in mapa:
    dupli.append(hg38)
  else:
    mapa[hg38] = ensg + ensemble

# Ciscenje mape od duplih
for gen in dupli:
  del mapa[gen]

# Izvestavanje o duplim ENSG-ovima
with open('../pomocno/dupli.txt', 'w') as izvestaj:
  dupli = '\n'.join(dupli)
  izvestaj.write(dupli)

# Serijalizacija mape
with open('../pomocno/hg38_ensg.json', 'w') as dat:
  json.dump(mapa, dat, indent=2)
