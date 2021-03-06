#### Klasterovanje
<img width="400" src="https://raw.githubusercontent.com/matfija/Klasterovanje-PBMC/master/pomocno/GSM333056x.png"> <img width="400" src="https://raw.githubusercontent.com/matfija/Klasterovanje-PBMC/master/pomocno/GSM333056x_4nk.png">

## Klasterovanje PBMC :page_facing_up:
Seminarski rad na kursu Istraživanje podataka 2. Cilj je bio formirati i uporediti različite modele klasterovanja za skup podataka koji predstavlja ljudske mononuklearne ćelije periferne krvi (PBMC) i time utvrditi i prošiti znanje iz nenadgledanog mašinskog učenja i rada sa retkim (sparse) i velikim podacima (big data).

Nakon kratke diskusije o samim podacima, primenjene su razne tehnike klasterovanja u Python-u uz biblioteke [sklearn](https://scikit-learn.org/stable/), [PyClustering](https://pyclustering.github.io/docs/0.8.2/html/index.html) i [MiniSom](https://github.com/JustGlowing/minisom), kao i specijalizovanoj aplikaciji [(g)CLUTO](http://glaros.dtc.umn.edu/gkhome/views/cluto). Neke od njih su metodi zasnovani na reprezentativnim tačkama (k-sredina, k-medijana, k-medoida), hijerarhijski metodi (sakupljajući i razdvajajući), metodi zasnovani na analizi gustine (DBSCAN, OPTICS), samoorganizujuće mape (SOM), spektralna analiza i mnoge druge, sve sa različitim merama rastojanja i drugim promenljivim parametrima. Za potpun uvid u primenjene metode, rezultate rada i donete zaključke pogledati priložene materijale i prateći dokument, dostupne na ovom repozitorijumu.

Po dogovoru sa mentorima, sami podaci nisu postavljeni na repozitorijum, ali izveštaji o njima jesu.
