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
import org.apache.spark.mllib.clustering.*;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.io.*;

/**
* Normalized Mutual Information is a measure of how informative is clustering
* against classes
*/
public class NMIRankedCategories {
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

        // path to a csv with tuples (category, importance score)
        String categoriesRankingPath = args[3];

        // load categories of each wikipedia article
        JavaRDD<WikiPage> pages = InputOutput.read(sc, datasetPath);

        // retrieve categories ranking csv
        JavaPairRDD<String, Double> categoriesRankingRDD =
          // read file as JavaRDD of strings
          sc.textFile(categoriesRankingPath)
          .mapToPair((row) -> {
            String[] chunks = row.split(",");

            try {
              // since there are commas in category names
              // only last one is used as a separator
              String category = chunks[0];
              for (int i = 1; i < chunks.length - 1; i++) {
                category += "," + chunks[i];
              }
              double ranking = Double.parseDouble(chunks[chunks.length - 1]);
              return new Tuple2<>(category, ranking);
            }
            catch (Exception e) {
              throw new Exception("Unable to parse line: " + row);
            }
          });

        Map<String, Double> categoriesRanking = categoriesRankingRDD.collectAsMap();
        sc.broadcast(categoriesRanking);

        // create category RDD with the most important category among each article
        JavaRDD<String> categoryRDD = pages
          .map((page) -> Arrays.asList(page.getCategories()))
          .map((categories) -> {
            String bestCategory = "";
            double bestScore = 0;

            for (String cat : categories) {
              double currentScore = categoriesRanking.get(cat);
              if (currentScore > bestScore) {
                bestCategory = cat;
                bestScore = currentScore;
              }
            }
            return bestCategory;
          })
          .cache();

        // load vector representation of each article
        JavaRDD<Tuple2<Long, Vector>> wikiVectorsDump = sc.objectFile(wikiVectorPath);
        JavaRDD<Vector> wikiVectors = wikiVectorsDump
          .map((row) -> row._2())
          .cache();

        // number of documents is computed after category filtering
        long numDocuments = categoryRDD.count();
        assert numDocuments == wikiVectors.count();

        // obtain number of occurences of each category, i.e. P(c) in NMI formula
        JavaPairRDD<String, Integer> categoryCountRDD = categoryRDD
          .mapToPair((category) -> new Tuple2<>(category, 1))
          .reduceByKey((x, y) -> x + y)
          .cache();

        Map<String, Integer> categoryCount = categoryCountRDD.collectAsMap();

        double categoriesEntropy = categoryCountRDD
          .map((row) -> row._2() / (double) numDocuments)
          .map((probability) -> - probability * Math.log(probability) / Math.log(2))
          .reduce((x, y) -> x + y);

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
          JavaRDD<Integer> clusterIDs;
          switch (clusteringTecnique) {
              case "KMeans":
                KMeansModel Kmodel = KMeansModel.load(sc.sc(), modelPath);
                clusterIDs = Kmodel.predict(wikiVectors);
                break;

              case "LDA":
                // LDA is not able to load models, nor to predict the cluster of a new document
                // so I will read (wikiPageID, clusterID) results CSV
                clusterIDs =
                  sc.textFile(modelPath)
                  .mapToPair((row) -> {
                    String[] chunks = row.split(",");
                    try {
                      return Integer.parseInt(chunks[chunks.length - 1]);
                    }
                    catch (Exception e) {
                      throw new Exception("Unable to parse line: " + row);
                    }
                  });
                break;
          }

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
              double probability = row._2() / (double) numDocuments;
              return - probability * Math.log(probability) / Math.log(2);
            })
            .reduce((x, y) -> x + y);

          // create a RDD with (categories, clusterID)
          JavaPairRDD<String, Integer> categoryID = categoryRDD.coalesce(1)
            .zip(clusterIDs.coalesce(1))
            .cache();

          // count couples (category, clusterID), i.e. P(w or C) probabilities
          JavaPairRDD<Tuple2<String, Integer>, Integer> categoryClusterCount =
            categoryID
            // map tuples to key, value tuple ( (clusterID, category) , 1)
            .mapToPair((row) -> {
              String category = row._1();
              int clusterID = row._2();
              return new Tuple2<>(new Tuple2<>(category, clusterID), 1);
            })
            // sum every element by key, to obtain counts of couples (clusterID, category)
            .reduceByKey((x, y) -> x + y);

          double modelScore = categoryClusterCount
            // compute term in NMI that involves a single category
            .map((row) -> {
              // extract tuple elements
              int clusterID = row._1()._2();
              String cat = row._1()._1();
              double Pwc = row._2() / (double) numDocuments;

              // retrieve precomputed probabilities
              double Pw = clusterCount.get(clusterID) / (double) numDocuments;

              /* what to do if category is not present */
              double Pc = categoryCount.get(cat) / (double) numDocuments;

              // sum over class and its complementary, to find
              // term in sum regarding current w and c
              double clusterCategoryScore;
              // deal with single cluster containing all memebers of a category
              if (Pw <= Pwc) {
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

              return clusterCategoryScore;
            })
            // sum across all (category, cluster) couples
            .reduce((x, y) -> x + y)
            // normalize to total entropy (clustering + classes)
            * 2 / (clusteringEntropy + categoriesEntropy);

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
          writer.write(line + "\n");
        }
        writer.close();
      }
      catch (IOException e) {
        System.err.println("Unable to write on file output/modelNMIscores.csv");
      }
    }
}
