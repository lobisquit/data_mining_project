Directory from different processing phases have different endind names.

- `.w2v` is the result of Word2VecModel saving to file from Word2VecFit class
- `.wpv` is the Object dump of `JavaRDD<Tuple2<Long, Vector>> wikiVectors` from Doc2Vec class
- `.cr` is the cluster result as JSON dump of `JavaRDD<Tuple2<Long, Vector>> wikiVectors` from Cluster class
