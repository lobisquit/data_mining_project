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
    String dataPath = args[0];

    // Usual setup
    SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("ERROR");

    // Load dataset of pages
    JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

    // Get text out of pages
    JavaRDD<String> texts = pages.map((p) -> p.getText());

    // Get the lemmas. It's better to cache this RDD since the
    // following operation, lemmatization, will go through it two
    // times.
    JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

    //word2vec
    Word2Vec w2vec=new Word2Vec();
    w2vec.setLearningRate(0.025);
    w2vec.setMaxSentenceLength(5000);
    w2vec.setMinCount(5);
    w2vec.setNumIterations(1);
    w2vec.setNumPartitions(1);
    w2vec.setVectorSize(100);
    w2vec.setWindowSize(5);
    Word2VecModel w2vM=w2vec.fit(lemmas);
    w2vM.save(JavaSparkContext.toSparkContext(sc),dataPath+".w2v");
  }
}
