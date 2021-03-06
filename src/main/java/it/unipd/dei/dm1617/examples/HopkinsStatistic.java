package it.unipd.dei.dm1617.examples;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Class to compute the Hopkins statistics.
 */
public class HopkinsStatistic {

    /**
     * @param args
     *  args[0] : path of the .wv files
     *  args[1] : fraction of the dataset used as sample for the Hopkins statistic
     *  args[2] : filename for the results (for example 'hopkins.csv')
     */
    public static void main(String[] args) {
        String path = args[0];
        if (!path.endsWith("/")) {
            path = path + "/";
        }

        double sampleFrac = Double.parseDouble(args[1]);

        String outputFile = args[2];

        // Usual setup
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // Load file
        System.out.println("Loading files..");
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

        // Needed RDD
        JavaRDD<Vector> onlyVectors = allWikiVector.map(elem -> elem._2());
        JavaRDD<Vector> dataSample = onlyVectors.sample(false, sampleFrac);
        JavaRDD<Vector> negativeDataSample = onlyVectors.subtract(dataSample);

        int vectorSize = onlyVectors.take(1).get(0).size(); // get the dimension of a single vector


        System.out.println("Computing vectors domain..");

        // Compute the max and min values for every vectors component
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

        // Generate a random set of vectors
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

        // Find the nearest sampled point for every vector (data subset and random sample)

        List<Vector> listNegativeDataSample = negativeDataSample.collect();
        // Kmeans model used with defined centers in order to compute distances
        KMeansModel dataCentersModel = new KMeansModel(listNegativeDataSample);
        JavaRDD<Double> dataDistances = dataSample.map((vector)->{
            Vector nearestPoint = listNegativeDataSample.get(dataCentersModel.predict(vector));
            return Vectors.sqdist(nearestPoint, vector);
        });

        List<Vector> listComplete = onlyVectors.collect();
        // kmeans model used with defined centers in order to compute distances
        KMeansModel randCentersModel = new KMeansModel(listComplete);
        JavaRDD<Double> randDistances = randSample.map((vector)->{
            Vector nearestPoint = listComplete.get(randCentersModel.predict(vector));
            return Vectors.sqdist(nearestPoint, vector);
        });

        System.out.println("Computing Hopkins statistics..");

        double dataDistSum = dataDistances.reduce((accum, n) -> (accum + n));
        double randDistSum = randDistances.reduce((accum, n) -> (accum + n));
        double hopkins = dataDistSum / (dataDistSum+randDistSum);
        System.out.println("dataDistSum = "+dataDistSum+", randDistSum = "+randDistSum);
        System.out.println("Hopkins statistic: "+hopkins);
        saveToFileAsCSV(onlyVectors.count(),sampleFrac,hopkins,outputFile);
    }

    /**
     * Writes the computation results in a csv file
     *
     * @param numVectors Number of vectors in the dataset
     * @param frac Fraction of the dataset used for the sample
     * @param hopkins Hopkins statistics estimated
     * @param fileName Filename for the results
     */
    private static void saveToFileAsCSV(long numVectors, double frac, double hopkins, String fileName){
        final String outputPath = "./output/";
        if (!fileName.endsWith(".csv")) {
            fileName = fileName.concat(".csv");
        }
        File directory = new File(outputPath);
        if (! directory.exists()){
            directory.mkdir();
        }
        File file = new File(outputPath + "/" + fileName);
        try{
            FileWriter filew = new FileWriter(file,true);
            if (!file.exists()) {
                filew.write("Dataset size,Sample fraction,Hopkins Statistics\n");
            }
            filew.write(numVectors + "," +frac + "," + hopkins +"\n");
            filew.close();
        }
        catch(IOException e){
            e.printStackTrace();
            System.exit(-1);
        }
    }
}