package it.unipd.dei.dm1617.examples;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;
import scala.util.parsing.json.JSONObject;

/**
 *
 * @author massimo
 */
public class Cluster {
    public static void main(String[] args){
        String path = args[0];

        // Usual setup
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");
        System.out.println("##############################################################################################################################");

        //todo cer
        //load file
        JavaRDD<Tuple2<Long, Vector>> wikiVectors = sc.objectFile(path);
        JavaRDD<Vector> prova = wikiVectors.map(elem->{
            return elem._2();
        });
        System.out.println("dimensione di prova "+prova.count());

        // Cluster the data into two classes using KMeans
        int numClusters = 60;
        int numIterations = 100;
        System.out.println("eseguo il kmeans");
        KMeansModel clusters = KMeans.train(prova.rdd(), numClusters, numIterations);
        /*System.out.println("Cluster centers:");
        for (Vector center: clusters.clusterCenters()) {
            System.out.println(" " + center);
        }*/

        double cost = clusters.computeCost(prova.rdd());
        System.out.println("Cost: " + cost);
    }
}
