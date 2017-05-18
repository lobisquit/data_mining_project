package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.List;

/**
 * Example program to show the basic usage of some Spark utilities.
 */
public class Word2VecFit {

  public static void main(String[] args) {
    String inputPath = args[0];

    // usual Spark setup
    SparkConf conf = new SparkConf(true).setAppName("Word2Vec");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("ERROR");

    // mark the starting point of our subsequent messages
    System.out.println("###################################################" +
        "#################################################################");

    // load dataset of wiki pages
    JavaRDD<WikiPage> pages = InputOutput.read(sc, inputPath);

    // extract text out of pages and pass it to the lemmatizer
    JavaRDD<String> texts = pages.map((p) -> p.getText());
    JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

    // perform word2vec
    Word2Vec w2vec=new Word2Vec()
        .setLearningRate(0.025)
        .setMaxSentenceLength(5000)
        .setMinCount(5)
        .setNumIterations(1)
        .setNumPartitions(1)
        .setVectorSize(100)
        .setWindowSize(5);

    // fit model and save to output file
    Word2VecModel w2vM = w2vec.fit(lemmas);
        w2vM.save(JavaSparkContext.toSparkContext(sc), "output/"+ inputPath.split("/")[1] + ".w2v");
  }
}
