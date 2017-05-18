package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.WikiPage;
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
/**
 * @param path of directory where file generated by Doc2Vec are saved
 */
public class Cluster {
    /**
     * @param args[0] path of medium-sample.dat.wpv
     * @param args[1] clustering tecnique name
     * @param args[2] number of cluster
     * @param args[3] number of iteration
     */
    public static void main(String[] args){
        // reading input path
        String path = args[0];
        if(!path.endsWith("/")){
            path=path + "/";
        }

        // reading clustering tecnique name
        String clusteringName = args[1];
        int numClusters = Integer.parseInt(args[2]);
        int numIterations = Integer.parseInt(args[3]);

        // usual Spark setup
        SparkConf conf = new SparkConf(true).setAppName("Clustering");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // mark the starting point of our subsequent messages
        System.out.println("###################################################" +
            "#################################################################");

        // load Doc2Vec page representation, i.e. tuples (wikipage_id, vector),
        // from the multiple output files
        System.out.println("load files");
        ArrayList<JavaRDD<Tuple2<Long, Vector>>> wikiVectors = new ArrayList();
        File folder = new File(path);
        for (File file : folder.listFiles()) {
            String fName=file.getName();
            if (file.isFile() && !fName.startsWith("_") && !fName.startsWith(".")) {
                wikiVectors.add(sc.objectFile(path + fName));
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
                KMeansModel kmeans =
                    KMeans.train(onlyVectors.rdd(), numClusters, numIterations);
                clusterIDs = kmeans.predict(onlyVectors);

                // compute kmeans objective function on training dataset
                System.out.println("-------------> kmeans objective function = "
                    + kmeans.computeCost(onlyVectors.rdd()));
                System.out.println("-------------> see https://goo.gl/QnjpHo");
                break;

            case "GaussianMixture":
                GaussianMixtureModel gaussianMixture =
                    new GaussianMixture()
                        .setK(numClusters)
                        .run(onlyVectors.rdd());
                clusterIDs = gaussianMixture.predict(onlyVectors);
                break;

            case "BisectingKMeans":
                BisectingKMeansModel bisectingKmeans = new BisectingKMeans()
                    .setK(numClusters)
                    .run(onlyVectors.rdd());
                clusterIDs = bisectingKmeans.predict(onlyVectors);

            default:
                throw new IllegalArgumentException(
                    "Invalid clustering tecnique -> " + clusteringName);
        }

        // create an RDD with (cluster_id, (wikipage_id, vector))
        JavaPairRDD<Integer, Tuple2<Long, Vector>> completeDataset =
            clusterIDs.zip(allWikiVector);

        // map each row to a json string representation, as a general save / load
        // format
        JavaRDD<String> jsonDataset = completeDataset.map((tuple) -> {
            return "{" +
                    "\"id_cluster\": " + tuple._1() + "," +
                    "\"id_wiki\": " + tuple._2()._1() + "," +
                    "\"vector\": " + tuple._2()._2()
                   + "}";
        });

        // collapse all parallel outputs to a single RDD (1) and save
        // this is needed to have a single output file
        jsonDataset.coalesce(1).saveAsTextFile(
            "output/cluster_" + clusteringName + ".cr");
    }

}
