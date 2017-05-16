package it.unipd.dei.dm1617.examples;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;
import scala.util.parsing.json.JSONObject;

/**
 *
 * @author massimo
 */
public class Cluster {
    public static void main(String[] args){
        String path = args[0];

        // Usual setup
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //todo come leggo il file?
        /*//load file
        JavaRDD<String> lines = sc.textFile(path);
        JavaRDD<Tuple2<Long, Vector>> wikiVectors = lines.map(line->{

            return
        });*/

    }
}
