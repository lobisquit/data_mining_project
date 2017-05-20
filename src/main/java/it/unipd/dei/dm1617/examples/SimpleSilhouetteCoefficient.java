package it.unipd.dei.dm1617.examples;

import org.apache.hadoop.mapred.join.ArrayListBackedIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.mllib.clustering.*;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;

import org.apache.spark.mllib.util.MLUtils;
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
        
        int kStart = new Integer.parseInt(args[0]);
        int kEnd = new Integer.parseInt(args[1]);

        // reading clustering tecnique name
        String dataset = "dataset/medium-sample.dat.wpv";
        String clusteringName = "KMeans";
        //int numClusters = 50;
        int numIterations = 20;

        // Spark setup
        // setMaster is needed to call the clustering (performed in Cluster.java) without conflicts
        SparkConf conf = new SparkConf(false).setAppName("SimplelSilhouetteCoefficient").setMaster("local[2]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // mark the starting point of our subsequent messages
        System.out.println("Starting Simple Silhouette");

        JavaRDD<Vector> articlesAsVectors = getArticlesAsVectors(dataset);
        System.out.println("Articles representation loaded");

        
        for(int i=kStart; i <= kEnd; i*=2){

            KMeansModel model = getKMeansModel(sc, dataset, clusteringName, i, numIterations);

            Vector[] centroids = model.clusterCenters();

            JavaRDD<Integer> predictedClusters = model.predict(articlesAsVectors);

            // computeDistanceFromItsCentroid
            //System.out.print("No");

            //MLUtils.fastSquaredDistance(centroids[0], norm1, centroids[1], double norm2, 5);

            // computeMinDistanceFromOtherCentroids
            
            // calculateSimpleSilhouetteCoefficient

        }        
    }

    public static KMeansModel getKMeansModel(JavaSparkContext sc, String dataset, String clusteringName, int numClusters, int numIterations){

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

        return kMeansModel;
    }

    public static JavaRDD<Vector> getArticlesAsVectors(String dataset){

        String wpvPath = dataset;
        if(!wpvPath.endsWith("/")){
            wpvPath = wpvPath + "/";
        }

        // load our articles represented as vectors
        // (wikipage_id, Doc2Vec vector)
        ArrayList<JavaRDD<Tuple2<Long, Vector>>> wikiVectors = new ArrayList();
        File folder = new File(wpvPath);
        for (File file : folder.listFiles()) {
            String fName = file.getName();
            if (file.isFile() && !fName.startsWith("_") && !fName.startsWith(".")) {
                wikiVectors.add(sc.objectFile(wpvPath + fName));
            }
        }

        // merge all chunks in  a single RDD
        JavaRDD<Tuple2<Long, Vector>> allWikiVector = wikiVectors.remove(0);
        for(JavaRDD<Tuple2<Long, Vector>> app:wikiVectors){
            allWikiVector = allWikiVector.union(app);
        }

        // remove id, since clustering requires RDD of Vectors
        JavaRDD<Vector> articlesAsVectors = allWikiVector.map(elem -> {
            return elem._2();
        });

        return articlesAsVectors;
    }

}
