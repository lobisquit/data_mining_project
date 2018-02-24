Data mining project, goal B
===========================
README italian version can be found [here](https://github.com/lobisquit/data_mining_project/blob/04b8595987cdc18d11fc6f0282c93f96df64db09/README.md).

Use Spark to cluster documents given their content and given Wikipedia categories.

Project structure
----------------------
Project is splitted into different folders.
- `dataset` contains input dataset
- `output` contains processing output dataset, such as intermediate computations
  of Spark
- `results` contains some `csv` reports used to make plots
- `latex` contains `tex` file of our report
- `src` contains code needed to perform computations and plot

Start software
--------------------
To make handling classes and parameter simpler, we wrote a `python` script,
namely `make.py`. Its documentation is simply given through `python3 make.py --help`.

Example usage is
```bash
python make.py --class Cluster (...args for Java main...)
```

Relevant classes with main Spark procedures can be found in
`src/main/java/it/unipd/dei/dm1617/examples/`, descripted in next section.

Spark classes
---------------------------

They are splitted in different groups, each one providing a differe processing
step. Parameters are retrievable in `main` method of each class.

- preprocessing
  - `CategoriesPreprocessing.java` counts articles per category
  - `TfidfCategories.java` ranks categories by their relevance
  - `TfIdf.java` builds `bag-of-words` model
  - `Word2VecFit.java` trains word2vec model using text corpus
  - `Doc2Vec.java` loads word2vec model and writes vector corresponding to each
     document in `output/` folder

- clustering
  - `Cluster.java` clusters input data and outputs the trained separation model

- result evaluation
  - `HopkinsStatistic.java` computes Hopkins statistic on vectorized corpus
  - `EvaluationLDA.java` inspect LDA fit output
  - `NMIRankedCategories.java` complutes NMI score considering one category per
    document only
  - `NMIOverlappingCategories.java` computes NMI score cosidering multiple
    categories per document
  - `SimpleSilhouetteCoefficient.java` complutes simple silhouette score given a
    `(vector, clusterID)` dataset

Script di servizio
------------------
In `src/` e `results/` can be found `python` scripts that process `output/`
files and build relevant plots.

`src/hierarchicalClustering.py` tried to use `scipy` clustering library, but it
was dropped given its RAM request.
