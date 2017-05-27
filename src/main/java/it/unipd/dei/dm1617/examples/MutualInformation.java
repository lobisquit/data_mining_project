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

        // load categories of each wikipedia article
        JavaRDD<WikiPage> pages = InputOutput.read(sc, datasetPath);

        // load vector representation of each article
        JavaRDD<Tuple2<Long, Vector>> wikiVectorsDump = sc.objectFile(wikiVectorPath);
        JavaRDD<Vector> wikiVectors = wikiVectorsDump
          .map((row) -> row._2())
          .cache();

        // obtain number of occurences of each category, i.e. P(c) in NMI formula
        JavaPairRDD<String, Integer> categoryCountRDD = pages
          // extract categories list from each page
          .map((page) -> Arrays.asList(page.getCategories()))
          // flatten all categories lists in a single RDD
          .flatMap((categories) -> categories.iterator())
          // count each occurrence of a category
          .mapToPair((category) -> new Tuple2<>(category, 1))
          .reduceByKey((x, y) -> x + y)
          // remove too unfrequent categories
          .filter((point) -> point._2() > 1)
          .cache();

        // retrieve a list of all categories in dataset
        List<String> categoriesSet = categoryCountRDD
          .map((row) -> row._1())
          .collect();
        sc.broadcast(categoriesSet);

        // create map (category, count) and broadcast it to workers
        Map<String, Integer> categoryCount = categoryCountRDD.collectAsMap();
        sc.broadcast(categoryCount);

        // with the knowledge acquired, remove all non-relevant categories
        // and the articles that remain without a category
        JavaPairRDD<List<String>, Vector> categoriesVector = pages
          // extract categories list from each page
          .map((page) -> Arrays.asList(page.getCategories()))
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

        // split categoriesVectors RDD in its columnsm since after that
        // they will be used separately
        JavaRDD<List<String>> categories = categoriesVector
          .map((row) -> row._1())
          .cache();
        JavaRDD<Vector> vectors = categoriesVector
          .map((row) -> row._2())
          .cache();

        // number of documents is computed after category filtering
        long numDocuments = categoriesVector.count();

        // retrieve model paths from proper directory
        File modelsFolder = new File("output/");
        List<String> modelPaths = new ArrayList();
        for (File modelPath : modelsFolder.listFiles()) {
          String modelPathStr = modelPath.toString();

          // select only models
          if (modelPathStr.endsWith(".cm")) {
            // select only models of wanted tecnique
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

          JavaRDD<Integer> clusterIDs = model.predict(vectors);

          // compute number of cluster elements and global entropy
          JavaPairRDD<Integer, Integer> clusterCountRDD = clusterIDs
            // count each cluster members
            .mapToPair((id) -> new Tuple2<>(id, 1))
            .reduceByKey((x, y) -> x + y)
            .cache();

          // values of P(w) are broadcasted
          Map<Integer, Integer> clusterCount = clusterCountRDD.collectAsMap();
          sc.broadcast(clusterCount);

          // compute entropy Hw of current cluster
          double clusteringEntropy = clusterCountRDD
            .map((row) -> {
              double probability = row._2() / numDocuments;
              return - probability * Math.log(probability) / Math.log(2);
            })
            .reduce((x, y) -> x + y);

          // create a RDD with (categories, clusterID)
          JavaPairRDD<List<String>, Integer> categoriesID = categories.coalesce(1)
            .zip(clusterIDs.coalesce(1))
            .cache();

          // create a RDD with (clusterID, single category)
          JavaPairRDD<Integer, String> categoryID = categoriesID
            .flatMap((point) -> {
                // create a collections of all tuples (clusterID, category)
                List<Tuple2<Integer, String>> points = new ArrayList();
                for (String category : point._1()) {
                    points.add(new Tuple2<>(point._2(), category));
                }
                return points.iterator();
            })
            // needed to create a JavaPairRDD
            .mapToPair((row) -> row);

          // count couples (category, clusterID), i.e. P(w or C) probabilities
          JavaPairRDD<Tuple2<Integer, String>, Integer> categoryClusterCount =
            categoryID
            // map tuples to key, value tuple ( (clusterID, category) , 1)
            .mapToPair((row) -> {
              String category = row._2();
              int clusterID = row._1();
              return new Tuple2<>(new Tuple2<>(clusterID, category), 1);
            })
            // sum every element by key, to obtain counts of couples (clusterID, category)
            .reduceByKey((x, y) -> x + y);

          double modelScore = categoryClusterCount
            // compute term in NMI that involves a single category
            .mapToPair((row) -> {
              // extract tuple elements
              int clusterID = row._1()._1();
              String cat = row._1()._2();
              double Pwc = row._2() / numDocuments;

              // retrieve precomputed probabilities
              double Pw = clusterCount.get(clusterID) / numDocuments;

              /* what to do if category is not present */
              double Pc = categoryCount.get(cat) / numDocuments;

              // TODO fix corner case of this bug
              if (row._2() > clusterCount.get(clusterID) ||
                  row._2() > categoryCount.get(cat)) {
                System.out.println(
                  "Cluster model " + modelPath +
                  "\nPwc= " + row._2() +
                  "\nPw=  " + clusterCount.get(clusterID) +
                  "\nPc=  " + categoryCount.get(cat) +
                  "\n-------------------------------------");
                return new Tuple2<>(cat, 0.0);
              }

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

              double categoryFraction = categoryCount.get(cat) / (double) numDocuments;
              double categoryEntropy = - categoryFraction * Math.log(categoryFraction) / Math.log(2);
              // normalize to total entropy (clustering + current class)
              return 2 * clusterCategoryScore /
                (clusteringEntropy + categoryEntropy);
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
