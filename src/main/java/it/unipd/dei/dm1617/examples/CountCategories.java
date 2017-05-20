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
 * Program that counts the categories.
 */
public class CountCategories {

    public static void main(String[] args) {
        String dataPath = args[0];

        SparkConf conf = new SparkConf(true).setAppName("Count Categories");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // useful to see your output
        sc.setLogLevel("ERROR");

        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

        JavaRDD<String> categories = pages.flatMap((doc) -> new ArrayListIterator(doc.getCategories()));

        JavaPairRDD<String, Integer> categoriesCounts = categories
                .mapToPair((w) -> new Tuple2<>(w, 1))
                .reduceByKey((x, y) -> x + y);

        System.out.println(categoriesCounts.take(10));

        // save result as text in proper directory
        categoriesCounts.saveAsTextFile("output/sort_and_count_categories/");

    }

}
