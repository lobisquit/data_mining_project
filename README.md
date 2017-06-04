Data mining project, goal B
===========================

Struttura del progetto
----------------------
Il progetto è composto da queste cartelle
- `dataset` contiene i dati di input
- `latex` contiene il file `tex` della nostra relazione
- `output` contiene i dataset di output delle nostre elaborazioni
- `results` contiene alcuni report in `csv` e gli script con cui sono state generate le figure
- `src` contiene il codice dell'elaborazione

Lanciare il software
--------------------
Per rendere più semplice la gestione delle classi e dei parametri abbiamo scritto uno
script `python`, ovvero `make.py` che automatizzi il processo, la cui documentazione si
ottiene con il comando `python3 make.py --help`.
Ad esempio
```bash
python make.py --class Cluster (...args for Java main...)
```

Le classi lanciate si trovano nel percorso `src/main/java/it/unipd/dei/dm1617/examples/`.
La descrizione dettagliata delle classi avverrà nella prossima sezione.

Classi che utilizzano Spark
---------------------------
Esse verranno divise a seconda della fase in cui sono utilizzate.
I parametri sono documentati nel metodo `main` delle rispettive classi.

- preprocessing
  - `CategoriesPreprocessing.java` conta gli articoli per categoria
  - `TfidfCategories.java` esegue un ranking tra le categorie per selezionare le più importanti
  - `TfIdf.java` costruisce il modello `bag-of-words`
  - `Word2VecFit.java` allena sul corpus di testi il modello word2vec
  - `Doc2Vec.java` carica il modello word2vec e salva in `output/` il vettore
     associato ad ogni articolo

- clustering
  - `Cluster.java` esegue il clustering dei  dati in input e salva in output il modello allenato

- valutazione dei risultati
  - `HopkinsStatistic.java` calcola la statistica di Hopkins del dataset vettorializzato
  - `EvaluationLDA.java` ispeziona il risultato del fit di LDA
  - `NMIRankedCategories.java` calcola il punteggio NMI considerando una sola categoria per articolo
  - `NMIOverlappingCategories.java` calcola il punteggio NMI considerando multiple categorie per articolo
  - `SimpleSilhouetteCoefficient.java` calcola simple silhouette dato un dataset `(vettore, clusterID)`

Script di servizio
------------------
In `src/` e `results/` sono presenti script `python` per compiere analisi sui file di `output/`
e per costruire gli opportuni grafici.

`src/hierarchicalClustering.py` è stato un tentativo di esecuire il clustering con
la libreria `scipy`, ma è stato abbandonato per l'enorme richiesta di RAM.
