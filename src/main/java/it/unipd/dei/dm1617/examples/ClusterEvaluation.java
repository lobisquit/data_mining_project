package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaPairRDD$;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.SparkSession;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.List;
import java.lang.Math;
import java.util.concurrent.ThreadLocalRandom;

/**
 *
 */
public class ClusterEvaluation {

    /** Fraction of the dataset used as sample for the Hopkins statistic */
    static final double SAMPLE_SIZE = 0.1;

    public static void main(String[] args) {
        String path = args[0];
        if (!path.endsWith("/")) {
            path = path + "/";
        }

        // Usual setup
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        //load file
        System.out.println("load files");
        ArrayList<JavaRDD<Tuple2<Long, Vector>>> wikiVectors = new ArrayList();
        File folder = new File(path);
        File[] listOfFiles = folder.listFiles();
        for (File file : listOfFiles) {
            String fName = file.getName();
            if (file.isFile() && !fName.startsWith("_") && !fName.startsWith(".")) {
                wikiVectors.add(sc.objectFile(path + "" + fName));
            }
        }

        System.out.println("get a unique file");
        JavaRDD<Tuple2<Long, Vector>> allWikiVector = wikiVectors.remove(0);
        for(JavaRDD<Tuple2<Long, Vector>> app:wikiVectors){
            allWikiVector = allWikiVector.union(app);
        }

        JavaRDD<Vector> onlyVectors = allWikiVector.map(elem->{
            return elem._2();
        });

        JavaRDD<Vector> dataSample = onlyVectors.sample(false, SAMPLE_SIZE);
        JavaRDD<Vector> negativeDataSample = onlyVectors.subtract(dataSample);

        int vectorSize = onlyVectors.take(1).get(0).size(); // get the dimension of a single vector


        System.out.println("Computing vectors domain..");

        // compute the max and min values for every vectors component
        Vector maxValues = onlyVectors.reduce((prev, n) -> {
            double[] pa = prev.toArray();
            double[] na = n.toArray();
            double[] res = new double[vectorSize];
            for (int i = 0; i < vectorSize; i++) {
                res[i] = (pa[i] < na[i]) ? na[i] : pa[i];
            }
            return new DenseVector(res);
        });
        Vector minValues = onlyVectors.reduce((prev, n) -> {
            double[] pa = prev.toArray();
            double[] na = n.toArray();
            double[] res = new double[vectorSize];
            for (int i = 0; i < vectorSize; i++) {
                res[i] = (pa[i] > na[i]) ? na[i] : pa[i];
            }
            return new DenseVector(res);
        });

        System.out.println("Generating random vectors..");

        // generate a random set of vectors
        long sampleNumber = dataSample.count();
        ArrayList<Vector> randList = new ArrayList<>();
        for (int i = 0; i < sampleNumber; i++) {
            double[] v = new double[vectorSize];
            for (int j = 0; j < vectorSize; j++) {
                v[j] = ThreadLocalRandom.current().nextDouble(minValues.apply(j), maxValues.apply(j));
            }
            randList.add(new DenseVector(v));
        }
        JavaRDD<Vector> randSample = sc.parallelize(randList);

        System.out.println("Computing distances..");

        // find the nearest sampled point for every vector

        List<Vector> listNegativeDataSample = negativeDataSample.collect();
        // kmeans model used with defined centers
        KMeansModel dataCentersModel = new KMeansModel(listNegativeDataSample);
        JavaRDD<Double> dataDistances = dataSample.map((vector)->{
            Vector nearestPoint = listNegativeDataSample.get(dataCentersModel.predict(vector));
            return Vectors.sqdist(nearestPoint, vector);
        });

        List<Vector> listComplete = onlyVectors.collect();
        // kmeans model used with defined centers
        KMeansModel randCentersModel = new KMeansModel(listComplete);
        JavaRDD<Double> randDistances = randSample.map((vector)->{
            Vector nearestPoint = listComplete.get(randCentersModel.predict(vector));
            return Vectors.sqdist(nearestPoint, vector);
        });

        System.out.println("Computing Hopkins statistics..");

        double dataDistSum = dataDistances.reduce((accum, n) -> (accum + n));
        System.out.println("1/2 reduce done.");
        double randDistSum = randDistances.reduce((accum, n) -> (accum + n));
        System.out.println("2/2 reduce done.");
        double hopkins = dataDistSum / (dataDistSum+randDistSum);
        System.out.println("dataDistSum = "+dataDistSum+", randDistSum = "+randDistSum);
        System.out.println("Hopkins statistic: "+hopkins);
        return;
    }
}