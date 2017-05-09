package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.commons.collections.iterators.ArrayListIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Program to count the categories.
 */
public class CategoriesPreprocessing {

  public static void main(String[] args) {
    String dataPath = args[0];

    // Usual setup
    SparkConf conf = new SparkConf(true).setAppName("Categories Count");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // useful to see your output
    sc.setLogLevel("ERROR");

    // Load dataset of pages
    JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

    JavaRDD<String> categories = pages.flatMap( (doc) -> new ArrayListIterator(doc.getCategories()) );

    JavaPairRDD<String, Integer> categoriesCount = categories
        .mapToPair((w) -> new Tuple2<>(w, 1 ))
        .reduceByKey((x, y) -> x + y);


    // List<Tuple2<WikiPage, Vector>> firstCount = categoriesCount.take(2);
    System.out.println("\n\n ######################## Categories Count \n");
    System.out.println( categoriesCount.take(1000 ) );
    System.out.println("\n\n # END ####################### \n");
    
  }

}
