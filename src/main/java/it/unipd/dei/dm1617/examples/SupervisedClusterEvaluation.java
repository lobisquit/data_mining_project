package it.unipd.dei.dm1617.examples;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
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

import java.util.ArrayList;
import java.io.File;

public class SupervisedClusterEvaluation {
    public static void main(String[] args){
        // usual Spark setup
        SparkConf conf = new SparkConf(true).setAppName("Clustering");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // mark the starting point of our subsequent messages
        System.out.println("###################################################" +
            "#################################################################");

        // path of folder with the output of Cluster.java
        String clusterModelPath = args[0];

        // path of original WikiPage dataset
        String datasetPath = args[1];

        // path of vector representation of wikipedia articles
        String wikiVectorPath = args[2];

        // load categories of each wikipedia article
        JavaRDD<WikiPage> pages = InputOutput.read(sc, datasetPath);

        JavaRDD<List<String>> categories = pages.map(
            (page) -> Arrays.asList(page.getCategories()));

        // load vector representation of each article
        JavaRDD<Tuple2<Long, Vector>> wikiVectors = sc.objectFile(wikiVectorPath);
        wikiVectors.cache();

        double numDocuments = (double) wikiVectors.count();

        // compute cluster of each input wikipage
        KMeansModel model = KMeansModel.load(sc.sc(), clusterModelPath);
        JavaRDD<Integer> clusterIDs = model.predict(
            wikiVectors.map((wikiVector) -> wikiVector._2()));

        // create a rdd (clusterID, category) for each category in every article
        JavaPairRDD<Integer, List<String>> dataset = clusterIDs.coalesce(1)
            .zip(categories.coalesce(1));

        JavaPairRDD<Integer, String> couples = JavaPairRDD.fromJavaRDD(
            dataset.flatMap((point) -> {
                List<Tuple2<Integer, String>> points = new ArrayList();
                for (String category : point._2()) {
                    points.add(new Tuple2<>(point._1(), category));
                }
                return points.iterator();
            }));

        // obtain dataset ((clusterID, category), count) and cache for performance
        JavaPairRDD<Tuple2<Integer, String>, Integer> couplesCounts = couples
            .mapToPair((couple)
                -> new Tuple2<Tuple2<Integer, String>, Integer>(couple, 1))
            .reduceByKey((x, y) -> x + y).cache();

        /* -----------------> compute clustering entropy Hw */

        // extract tuples (clusterID, count), to count members of each cluster
        JavaPairRDD<Integer, Integer> clusterCounts = couplesCounts
            .mapToPair((row) -> new Tuple2<>(row._1()._1(), row._2()))
            .reduceByKey((x, y) -> (x + y)).cache();

        // compute entropy of each cluster
        double Hw = - clusterCounts
            .map((row) -> {
                long pointPerCluster = row._2();
                return pointPerCluster / numDocuments;
            })
            .reduce((x, y) -> x + y);

        /* -----------------> compute categories (total) entropy Hw */

        // extract tuples (category, count), to count members of each category
        JavaPairRDD<String, Integer> catecoryCounts = couplesCounts
            .mapToPair((row) -> new Tuple2<>(row._1()._2(), row._2()))
            .reduceByKey((x, y) -> (x + y)).cache();

        double Hc = - catecoryCounts
            .map((row) -> {
                long pointPerCategory = row._2();
                return pointPerCategory / numDocuments;
            })
            .reduce((x, y) -> x + y);

        /* -----------------> create matrix (Pw, Pc) for each cluster w and category c */

        Map<String, Integer> categoryCountsMap = catecoryCounts.collectAsMap();
        Map<Integer, Integer> clusterCountsMap = clusterCounts.collectAsMap();

        double Hcw = - 2 / (Hc + Hw) * couplesCounts
            .map((row) -> {
                double Pc = categoryCountsMap.get(row._1()._2()) / numDocuments;
                double Pw = clusterCountsMap.get(row._1()._1()) / numDocuments;
                double Pwc = row._2() / numDocuments;
                return Pwc * Math.log(Pwc / (Pw * Pc));
            })
            .reduce((x, y) -> (x + y));

        // print normalized mutual information of clustering against classes
        System.out.println("" + Hcw);
    }

}
