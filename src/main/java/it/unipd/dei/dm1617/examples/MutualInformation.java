package it.unipd.dei.dm1617.examples;

import java.util.*;
import it.unipd.dei.dm1617.WikiPage;
import it.unipd.dei.dm1617.InputOutput;
import it.unipd.dei.dm1617.WikiV;
import org.apache.hadoop.mapred.join.ArrayListBackedIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.mllib.clustering.*;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.io.*;

/**
* Normalized Mutual Information is a measure of how informative is clustering
* against classes
*/
public class MutualInformation {
    public static void main(String[] args){
        // usual Spark setup
        SparkConf conf = new SparkConf(true).setAppName("Clustering");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // mark the starting point of our subsequent messages
        System.out.println("###################################################" +
            "#################################################################");

        // clustering tecnique to analize, as specified in
        // switch / case in Cluster.java
        String clusteringTecnique = args[0];

        // path of original WikiPage dataset
        String datasetPath = args[1];

        // path of vector representation of wikipedia articles
        String wikiVectorPath = args[2];

        /* -------------> create collection of (categories, wiki vector) for each article */

        // load categories of each wikipedia article
        JavaRDD<WikiPage> pages = InputOutput.read(sc, datasetPath);
        JavaRDD<List<String>> categories = pages
          .map((page) -> Arrays.asList(page.getCategories()));

        // load vector representation of each article
        JavaRDD<Tuple2<Long, Vector>> wikiVectorsDump = sc.objectFile(wikiVectorPath);
        JavaRDD<Vector> wikiVectors = wikiVectorsDump.map((row) -> row._2());

        /* -------------> preprocess categories, excluding ones with too few articles */

        // obtain number of occurences of each category, i.e. P(c) in NMI formula
        JavaPairRDD<String, Integer> categoriesCountsRDD = pages
          // obtain all categories in a single RDD
          .flatMap((page) -> Arrays.asList(page.getCategories()).iterator())
          // set every row frequency to 1
          .mapToPair((category) -> new Tuple2<>(category, 1))
          // convert to (key: value) RDD (JavaPairRDD) and sum occurences
          .reduceByKey((x, y) -> x + y)
          // remove too unfrequent categories
          .filter((point) -> point._2() > 1)
          .cache();

        // number of documents is computed after category filtering
        long numDocuments = categoriesCountsRDD.count();

        JavaPairRDD<String, Double> categoriesFractionsRDD = categoriesCountsRDD
          // normalize each count with total number of documents
          .mapToPair((row) ->
            new Tuple2<String, Double>(
              row._1(),
              row._2().doubleValue() / numDocuments))
          .cache();

        // create map (category, frequency) and broadcast it to workers
        Map<String, Double> categoriesFractions = categoriesFractionsRDD.collectAsMap();
        sc.broadcast(categoriesFractions);

        // compute entropy of indicator random variables corresponding to each category
        Map<String, Double> categoriesEntropies = categoriesFractionsRDD
          .mapToPair((row) -> {
            double probability = row._2();
            return new Tuple2<>(
              row._1(),
              - probability * Math.log(probability) / Math.log(2));
          }).collectAsMap();
        sc.broadcast(categoriesEntropies);

        // retrieve a list of all categories in dataset
        List<String> categoriesSet = categoriesFractionsRDD
          .map((row) -> row._1())
          .collect();
        sc.broadcast(categoriesSet);

        /* ----------> filter out too unfrequent categories */

        JavaPairRDD<List<String>, Vector> categoriesVectors = categories
          // create dataset with tuples (categories, wiki vector)
          .coalesce(1).zip(wikiVectors.coalesce(1))
          // removed previously filtered categories
          .mapToPair((row) -> {
            List<String> cats = row._1();
            Vector vector = row._2();

            List<String> filteredCategories = new ArrayList();
            for (String cat : cats) {
              if (categoriesSet.contains(cat)) {
                filteredCategories.add(cat);
              }
            }

            return new Tuple2<>(filteredCategories, vector);
          })
          // keep only articles with more than 0 categories
          .filter((row) -> !row._1().isEmpty())
          .cache();

        /* ----------> retrieve model paths from directory */

        File modelsFolder = new File("output/");
        List<String> modelPaths = new ArrayList();
        for (File modelPath : modelsFolder.listFiles()) {
          String modelPathStr = modelPath.toString();

          // select only models
          if (modelPathStr.endsWith(".cm")) {
            // select only model of wanted tecnique
            if (modelPathStr.split("/")[1].startsWith(clusteringTecnique)) {
              modelPaths.add(modelPathStr);
            }
          }
        }
        Collections.sort(modelPaths);

        // save results here for each model
        List<String> results = new ArrayList();

        // repeat procedure for each model
        for (String modelPath : modelPaths) {
          // compute clusterID of each input wikipage
          KMeansModel model = KMeansModel.load(sc.sc(), modelPath);

          JavaRDD<Integer> clusterIDs = model.predict(
            // exctract filtered vectors
            categoriesVectors.map((row) -> row._2()));

          /* ----------> compute clustering probabilities, i.e. P(w) */

          JavaPairRDD<Integer, Double> clusterFractionsRDD = clusterIDs
            // count cluster members
            .mapToPair((id) -> new Tuple2<>(id, 1))
            .reduceByKey((x, y) -> x + y)
            // normalize with number of documents
            .mapToPair((row) ->
              new Tuple2<>(
                row._1(),
                row._2().doubleValue() / numDocuments))
            .cache();

          // values of P(w) are broadcasted and clustering entropy Hw is precomputed
          Map<Integer, Double> clusterFractions = clusterFractionsRDD.collectAsMap();
          sc.broadcast(clusterFractions);

          double clusteringEntropy = clusterFractionsRDD
            .map((row) -> {
              double probability = row._2();
              return - probability * Math.log(probability) / Math.log(2);
            })
            .reduce((x, y) -> x + y);

          /* ----------> create a RDD with (category, clusterID, count) */

          // create a RDD with (clusterID, categories)
          JavaPairRDD<Integer, List<String>> categoriesID = clusterIDs.coalesce(1)
            .zip(categoriesVectors.map((row) -> row._1()).coalesce(1))
            .cache();

          // create a RDD with (clustedID, single category)
          JavaPairRDD<Integer, String> categoryIDcouples = categoriesID
            .flatMap((point) -> {
                // create a collections of all tuples (clusterID, category)
                List<Tuple2<Integer, String>> points = new ArrayList();
                for (String category : point._2()) {
                    points.add(new Tuple2<>(point._1(), category));
                }
                return points.iterator();
            })
            // needed to create a JavaPairRDD
            .mapToPair((row) -> row);

          // count couples (clusterID, category), i.e. P(w ∪ C) probabilities
          JavaPairRDD<Tuple2<Integer, String>, Double> categoriesClusterFractions =
            categoryIDcouples
            // map tuples to key, value tuple ( (clusterID, category) , 1)
            .mapToPair((row) ->
              new Tuple2<>(new Tuple2<>(row._1(), row._2()), 1))
            // sum every element by key, to obtain counts of couples (clusterID, category)
            .reduceByKey((x, y) -> x + y)
            // normalize count to obtain probabilities
            .mapToPair((row) -> {
                Tuple2<Integer, String> key = row._1();
                double fraction = row._2().doubleValue() / numDocuments;
                return new Tuple2<>(key, fraction);
              });

          double modelScore = categoriesClusterFractions
            // compute term in NMI that involves a single category
            .mapToPair((row) -> {
              // extract tuple elements
              int clusterID = row._1()._1();
              String cat = row._1()._2();
              double Pwc = row._2();

              // retrieve precomputed probabilities
              double Pw = clusterFractions.get(clusterID);

              /* what to do if category is not present */
              double Pc = categoriesFractions.get(cat);

              // sum over class and its complementary, to find
              // term in sum regarding current w and c
              double clusterCategoryScore;

              // deal with single cluster containing all memebers of a category
              if (Pw - Pwc == 0) {
                clusterCategoryScore =
                // note that a 0-probability event has entropy 0
               (Pwc * Math.log(Pwc / (Pc * Pw)) + 0) / Math.log(2);
              }
              else {
                clusterCategoryScore = (
                    Pwc * Math.log(Pwc / (Pc * Pw)) +
                    (Pw - Pwc) * Math.log((Pw - Pwc) / (Pw * (1 - Pc)))
                  ) / Math.log(2);
              }

              return new Tuple2<>(cat, clusterCategoryScore);
            })
            // sum over all clusters, to obtain each category score
            // (i.e. NMI numerator for each category)
            .reduceByKey((x, y) -> x + y)
            // normalize each category term
            .map((row) -> {
              String cat = row._1();
              double clusterCategoryScore = row._2();

              // normalize to total entropy (clustering + current class)
              return 2 * clusterCategoryScore /
                (clusteringEntropy + categoriesEntropies.get(cat));
            })
            // sum across all categories
            .reduce((x, y) -> x + y);

          // System.out.println(categoriesClusterFractions
          //   .filter((point) -> point._2() > 1/numDocuments)
          //   .take(10));
          System.out.println("Score for model " + modelPath + " = " + modelScore);
          results.add(modelPath + "," + modelScore);
        }
      // output results to csv file
      try {
        FileWriter writer = new FileWriter("output/modelNMIscores.csv");
        for(String line : results) {
          writer.write(line);
        }
        writer.close();
      }
      catch (IOException e) {
        System.err.println("Unable to write on file output/modelNMIscores.csv");
      }
    }
}
