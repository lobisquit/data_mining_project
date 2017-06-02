package it.unipd.dei.dm1617.examples;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

import java.io.FileWriter;
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
        
        int kStart = Integer.parseInt(args[0]);
        int kEnd = Integer.parseInt(args[1]);
        int kStep = Integer.parseInt(args[2]);

        //Integer[] listOfModels = {700, 989, 1111, 1571, 1763, 2494, 2800, 3959, 4444, 6285, 7055, 9978, 11200, 15839, 17778, 25143, 28222, 39912, 44800, 63356};
        // Integer[] listOfModels = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100};
        Integer[] listOfModels = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199};
        kStart = 0;
        kEnd = listOfModels.length - 1;
        kStep = 1;
        
        // set some parameters
        String dataset = "dataset/medium-sample.dat.wpv";
        String clusteringName = "KMeans";
        int numIterations = 30;

        ArrayList<Tuple2<Integer, Double>> results = new ArrayList<Tuple2<Integer, Double>>();

        // Spark setup
        // setMaster is needed to call the clustering (performed in Cluster.java) without conflicts
        SparkConf conf = new SparkConf(false).setAppName("SimplelSilhouetteCoefficient").setMaster("local[4]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // mark the starting point of our subsequent messages
        System.out.println("Starting Simple Silhouette");

        JavaRDD<Tuple2<Long, Vector>> articlesAsVectors = getArticlesAsVectors(sc, dataset);
        System.out.println("Articles representation loaded");
        
        for(Integer i= kStart; i <= kEnd; i += kStep){

            System.out.println("Computing kmeans with k="+listOfModels[i].toString()); // i

            KMeansModel model = getKMeansModel(sc, dataset, clusteringName, listOfModels[i], numIterations); // i

            Vector[] centroids = model.clusterCenters();

            sc.broadcast(centroids);

            JavaRDD<Integer> predictedClusters = model.predict(getOnlyVectors(articlesAsVectors));

            JavaPairRDD<Integer, Tuple2<Long, Vector>> zipped = predictedClusters.zip(articlesAsVectors);

            // pair: articleVector, clusterId
            JavaPairRDD<Vector, Integer> articles = zipped.mapToPair(pair -> new Tuple2<>(pair._2._2, pair._1));

            // computeDistanceFromItsCentroid
            JavaRDD<Double> simpleSilhoutteCoefficients = articles.map( pair -> {

                double distanceFromItsCentroid = Double.POSITIVE_INFINITY;
                double minDistanceFromOtherCentroids = Double.POSITIVE_INFINITY;

                for(int j=0; j < centroids.length; j++){

                    double euclideanDistance = Vectors.sqdist(pair._1, centroids[j]);

                    if(j == pair._2){ // its centroid
                        distanceFromItsCentroid = euclideanDistance;
                    }
                    else{
                        // keep track of the minimum
                        if(minDistanceFromOtherCentroids > euclideanDistance){
                            minDistanceFromOtherCentroids = euclideanDistance;
                        }
                    }
                }

                if(distanceFromItsCentroid == Double.POSITIVE_INFINITY ||
                        minDistanceFromOtherCentroids == Double.POSITIVE_INFINITY){
                    // something bad happened
                    // e.g. there is no centroid for cluster of this point
                    // or there are no other clusters centroids
                    throw new Exception("Bad! Either there is no centroid for this cluster or there are no other clusters (k=1 maybe?)");
                }

                // compute the Simple Silhouette Coefficient
                double simpleSilhouetteCoefficient =
                        (minDistanceFromOtherCentroids - distanceFromItsCentroid) / minDistanceFromOtherCentroids;

                if(distanceFromItsCentroid == 0.0)
                    return 0.0;

                return simpleSilhouetteCoefficient;
            } );

            Double sumCoefficients = simpleSilhoutteCoefficients.reduce(
                    (v1, v2) -> {
                        return v1 + v2;
                    }
            );
            
            Double ssc = sumCoefficients / new Long(simpleSilhoutteCoefficients.count()).doubleValue();
            
            /*Long totalCount = simpleSilhoutteCoefficients.filter(n -> {
                return n != 0.0;
            }).count();
            Double ssc = sumCoefficients / totalCount.doubleValue(); */

            results.add(new Tuple2<>(listOfModels[i], ssc)); // i

            System.out.println("Simple Silhouette Coefficient: " + ssc + " with K="+listOfModels[i].toString()); // i
            System.out.println("Sum: " + sumCoefficients + "totalCount: " + totalCount);

        }

        System.out.println("Saving to output file");
        // saves: k of clusters, simple silhouette coefficient
        saveToFileAsCSV(results);
        System.out.println("Done");
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

        System.out.println("Model computed");

        // load kmeansmodel representation
        KMeansModel kMeansModel = KMeansModel.load(sc.sc(), modelToLoad);

        return kMeansModel;
    }

    public static JavaRDD<Tuple2<Long, Vector>> getArticlesAsVectors(JavaSparkContext sc, String dataset){

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
        for(JavaRDD<Tuple2<Long, Vector>> app:wikiVectors) {
            allWikiVector = allWikiVector.union(app);
        }
        return allWikiVector;
    }

    public static void saveToFileAsCSV(ArrayList<Tuple2<Integer, Double>> tuples){

        try{
            FileWriter file = new FileWriter("./output/kSimpleSilhouette.csv");

            for (Tuple2<Integer, Double> tup : tuples) {
                file.write("" + tup._1 + ", " + tup._2 + "\n");
            }
            file.close();
        }
        catch(Exception e){
            System.out.println("Failed to write to disk");
            System.out.println(e);
            System.exit(1);
        }
    }

    public static JavaRDD<Vector> getOnlyVectors(JavaRDD<Tuple2<Long, Vector>> wikiVectors){
        // remove id, since clustering requires RDD of Vectors
        JavaRDD<Vector> onlyVectors = wikiVectors.map(elem -> {
            return elem._2();
        });
        return onlyVectors;
    }

}