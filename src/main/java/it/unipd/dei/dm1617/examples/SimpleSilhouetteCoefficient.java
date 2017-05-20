package it.unipd.dei.dm1617.examples;

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
/**
 * @param to do
 */
public class SimpleSilhouetteCoefficient {
    /**
     * @param args[0] number of clusters to start from
     * @param args[1] number of clusters to end to
     */
    public static void main(String[] args){
        
        String kStart = new Integer.parseInt(args[0]);
        String kEnd = new Integer.parseInt(args[1]);

        // reading clustering tecnique name
        String dataset = "dataset/medium-sample.dat.wpv";
        String clusteringName = "KMeans";
        int numClusters = 50;
        int numIterations = 20;

        // Spark setup
        // setMaster is needed to call the clustering (performed in Cluster.java) without conflicts
        SparkConf conf = new SparkConf(false).setAppName("SimplelSilhouetteCoefficient").setMaster("local[2]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // mark the starting point of our subsequent messages
        System.out.println("Starting Simple Silhouette");

        for(int i=kStart, i <= kEnd, i*=2){

            Vector[] centroids = loadCentroids(sc, clusteringName, numClusters, numIterations);

            // computeDistanceFromItsCentroid

            // computeMinDistanceFromOtherCentroids
            
            // calculateSimpleSilhouetteCoefficient

        }        
    }

    public static Vector[] loadCentroids(JavaSparkContext sc, String clusteringName, int numClusters, int numIterations){

        // uses output file with the format from Cluster.java class
        String modelToLoad =    "output/" + clusteringName +
                                "_n_cluster_" + numClusters +
                                "_n_iterat_" + numIterations +
                                ".cm";

        // if the kmeans model has not already been computed, lets do it
        if(!new File(modelToLoad).exists()){
            System.out.println("KMeansModel not found for " + numClusters + " clusters with "+ numIterations + " iterations.");
            System.out.println("Calculating model...");
            Cluster.doClustering(sc, dataset, clusteringName, numClusters, numIterations);
        }

        // load kmeansmodel representation
        KMeansModel kMeansModel = KMeansModel.load(sc.sc(), modelToLoad);
        Vector[] centroids = kMeansModel.clusterCenters();

        return centroids
    }

}
