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
import java.util.HashMap;

public class TfidfCategories {

    public static void main(String[] args) {
        String dataPath = args[0];

        // Usual setup
        SparkConf conf = new SparkConf(true).setAppName("Categories Count");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // useful to see your output
        sc.setLogLevel("ERROR");
        System.err.println("######################################################");
        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);


        //get all categories in a file
        JavaRDD<String> categories = pages.map(elem->{
            String str="";
            for(String s:elem.getCategories()) {
                str = str + "##" + s;
            }
            return str;
        });
        List<String> list=categories.collect();
        HashMap<String,Integer> mappa=new HashMap();
        int i=0;
        for(String line:list){
            String[] parts=line.trim().split("##");
            for(String cat:parts){
                if(!mappa.containsKey(cat)){
                    mappa.put(cat,i++);
                }
            }
        }
        System.err.println("fine generazione della mappa");
        System.err.println("genero la lista delle categorie nella pagina con le chiavi della mappa");
        //

        for(String line:list){
            String[] parts=line.trim().split("##");
            for(String cat:parts){
                Integer index=mappa.get(cat);
                int index2=index.intValue();
            }
        }
    }

}
