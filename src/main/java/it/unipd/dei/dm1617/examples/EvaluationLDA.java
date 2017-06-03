package it.unipd.dei.dm1617.examples;

import net.sf.cglib.core.Local;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.clustering.LDAModel;
import org.apache.spark.mllib.clustering.LocalLDAModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;
import scala.Tuple3;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/**
 * Class to inspect the clustering performed using the Latent Dirichlet Allocation
 */
public class EvaluationLDA {
    /**
     * @param args
     *  args[0] : path of the LDA model
     *  args[1] : path of the vocabulary
     */
    public static void main(String[] args) {
        String modelPath = args[0];
        String vocabPath = args[1];

        if (!modelPath.endsWith("/")) {
            modelPath = modelPath + "/";
        }

        // usual Spark setup
        SparkConf conf = new SparkConf(true).setAppName("Clustering");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        String[] lemmas = new String[3000];
        try {
            BufferedReader br = new BufferedReader(new FileReader(vocabPath));
            String line;
            int i = 0;
            while ((line = br.readLine()) != null) {
                lemmas[i] = line;
                i++;
            }
            br.close();
        } catch (Exception e) {
            System.out.println("Problems reading the vocabulary file");
            e.printStackTrace();
        }

        System.out.println("##############################################################");

        DistributedLDAModel model = DistributedLDAModel.load(sc.sc(), modelPath);

        double k = model.k(); // number of topics
        System.out.println("The number of topics is: "+k);

        double vocabSize = model.vocabSize();
        System.out.println("The vocabulary size is: "+vocabSize);

        double logLikelihood = model.logLikelihood();
        System.out.println("The logLikelihood size is: "+logLikelihood);

        // Print some topics together with their most characterizing lemmas
        Tuple2<int[], double[]>[] describeTopics = model.describeTopics(10);

        try{
            PrintWriter writer = new PrintWriter("./output/LDA-topicsdescription-k"+k+"-vocab3000.txt", "UTF-8");
            for (int i = 0; i < k; i++) {
                Tuple2<int[], double[]> topic = describeTopics[i];
                writer.print("Topic " + i + ":");
                for (int j = 0; j < topic._1.length; j++) {
                    writer.print(" " + lemmas[topic._1[j]]);
                }
                writer.print("\n");
            }
            writer.close();
        } catch (IOException e) {
            System.err.println("Error writing the topics descriptions");
            e.printStackTrace();
        }

        JavaRDD<Tuple3<Long, int[], int[]>> topicAssignments = model.javaTopicAssignments();
        List<Tuple3<Long, int[], int[]>> collection = topicAssignments.collect();
        try{
            PrintWriter writer = new PrintWriter("./output/LDA-assignement-k"+k+"-vocab3000.csv", "UTF-8");
            writer.println("ArticleID,ClusterID");
            for (int i = 0; i < collection.size(); i++) {
                Tuple3<Long, int[], int[]> tuple = collection.get(i);
                writer.println(tuple._1()+","+tuple._3()[0]);
            }
            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
        try{
            PrintWriter writer = new PrintWriter("./output/LDA-loglikelihood-k"+k+"-vocab3000.csv", "UTF-8");
            writer.println(""+logLikelihood);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
