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
     * @param args[0] path of medium-sample.dat.wpv
     * @param args[1] clustering tecnique name
     * @param args[2] number of cluster
     * @param args[3] number of iteration
     */
    public static void main(String[] args){
        

        // reading clustering tecnique name
        String dataset = args[0];
        //String clusteringName = args[1];
        String clusteringName = "KMeans";
        //int numClusters = Integer.parseInt(args[2]);
        int numClusters = 50;
        //int numIterations = Integer.parseInt(args[3]);
        int numIterations = 20;

        // usual Spark setup
        SparkConf conf = new SparkConf(false).setAppName("SimplelSilhouetteCoefficient");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // mark the starting point of our subsequent messages
        System.out.println("###");

        // uses output file with the format from Cluster.java class
        String modelToLoad =    "output/" + clusteringName +
                                "_n_cluster_" + numClusters +
                                "_n_iterat_" + numIterations +
                                ".cm";

        Cluster.doClustering(sc, dataset, clusteringName, numClusters, numIterations);

        // load kmeansmodel representation
        KMeansModel kMeansModel = KMeansModel.load(sc.sc(), modelToLoad);
        Vector[] centroids = kMeansModel.clusterCenters();

        System.out.print(centroids[0]);

        /*
        System.out.println("load files");
        ArrayList<JavaRDD<Tuple2<Long, Vector>>> wikiVectors = new ArrayList();
        File folder = new File(wpvPath);
        for (File file : folder.listFiles()) {
            String fName = file.getName();
            if (file.isFile() && !fName.startsWith("_") && !fName.startsWith(".")) {
                wikiVectors.add(sc.objectFile(wpvPath + fName));
            }
        }

        // merge all chunks in  a single RDD
        System.out.println("get a unique file");
        JavaRDD<Tuple2<Long, Vector>> allWikiVector = wikiVectors.remove(0);
        for(JavaRDD<Tuple2<Long, Vector>> app:wikiVectors){
            allWikiVector = allWikiVector.union(app);
        }

        // remove id, since clustering requires RDD of Vectors
        JavaRDD<Vector> onlyVectors = allWikiVector.map(elem -> {
            return elem._2();
        });

        // cluster the data into two classes using method specified in args[1]
        System.out.println("performing clustering");

        // train and classify dataset with the specified tecnique
        // note that corresponding group for each point of the input (training)
        // dataset is computed, to associate each cluster with the actual WikiPages
        // that it contains
        JavaRDD<Integer> clusterIDs;
        switch (clusteringName) {
            case "KMeans":
                // train model on dataset
                KMeansModel kmeans =
                    KMeans.train(onlyVectors.rdd(), numClusters, numIterations);

                // save model to output and exit
                kmeans.save(sc.sc(),
                    "output/" + clusteringName +
                    "_n_cluster_" + numClusters +
                    "_n_iterat_" + numIterations +
                    ".cm");
                break;

            case "GaussianMixture":
                GaussianMixtureModel gaussianMixture =
                    new GaussianMixture()
                        .setK(numClusters)
                        .run(onlyVectors.rdd());
                gaussianMixture.save(sc.sc(),
                    "output/" + clusteringName +
                    "_n_cluster_" + numClusters +
                    "_n_iterat_" + numIterations +
                    ".cm");
                break;

            case "BisectingKMeans":
                BisectingKMeansModel bisectingKmeans = new BisectingKMeans()
                    .setK(numClusters)
                    .run(onlyVectors.rdd());
                bisectingKmeans.save(sc.sc(),
                    "output/" + clusteringName +
                    "_n_cluster_" + numClusters +
                    "_n_iterat_" + numIterations +
                    ".cm");
                break;

            default:
                throw new IllegalArgumentException(
                    "Invalid clustering tecnique -> " + clusteringName);
        }/**/
    }

}
