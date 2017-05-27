package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.commons.collections.iterators.ArrayListIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.SparkSession;
import scala.Float;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.HashMap;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class TfidfCategories {

    public static void main(String[] args) {
        String dataPath = args[0];

        // Usual setup
        SparkConf conf = new SparkConf(true).setAppName("Categories Count");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");
        System.err.println("######################################################");

        // Load dataset of wikipedia pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

        System.err.println("load categories");
        //get all categories in a List
        JavaRDD<List<String>> categories = pages.map(elem->{
            List<String> ret = new ArrayList<String>();
            for(String str:elem.getCategories()){
                ret.add(str);
            }
            return ret;
        });
        List<List<String>> articlesCategories = categories.collect();

        //collect categories in one only List
        System.err.println("generate categoriesUnique");
        int cont = 0;
        List<String> categoriesUnique=new ArrayList<String>();
        for(List<String> art:articlesCategories){
            for(String cat:art){
                if(!categoriesUnique.contains(cat)){
                    cont++;
                    categoriesUnique.add(cat);
                    if(cont%1000==0)
                        System.err.println("-> "+cont);
                }
            }
        }

        System.err.println("caclulate idf foreach categories");
        //calculate idf foreach categories and save the score in a file
        TfidfCategories calculator = new TfidfCategories();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter("output/categoriesRanking.csv"))) {
            cont=0;
            for(String cat:categoriesUnique) {
                System.err.println(" -> "+(((float)cont)/categoriesUnique.size()*100)+" %");
                bw.write(cat+","+calculator.idf(articlesCategories, cat)+"\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    /**
     * @param doc  list of strings
     * @param term String represents a term
     * @return term frequency of term in document
     */
    public double tf(List<String> doc, String term) {
        double result = 0;
        for (String word : doc) {
            if (term.equalsIgnoreCase(word))
                result++;
        }
        return result / doc.size();
    }

    /**
     * @param docs list of list of strings represents the dataset
     * @param term String represents a term
     * @return the inverse term frequency of term in documents
     */
    public double idf(List<List<String>> docs, String term) {
        double n = 0;
        for (List<String> doc : docs) {
            for (String word : doc) {
                if (term.equalsIgnoreCase(word)) {
                    n++;
                    break;
                }
            }
        }
        return Math.log(docs.size() / n);
    }

    /**
     * @param doc  a text document
     * @param docs all documents
     * @param term term
     * @return the TF-IDF of term
     */
    public double tfIdf(List<String> doc, List<List<String>> docs, String term) {
        return tf(doc, term) * idf(docs, term);

    }
    /*
    public static void main(String[] args) {


        TFIDFCalculator calculator = new TFIDFCalculator();
        System.out.println("lorem "+calculator.tfIdf(doc1, documents, "lorem"));
        System.out.println("lorem "+calculator.tfIdf(doc2, documents, "lorem"));
        System.out.println("ipsum "+calculator.tfIdf(doc1, documents, "ipsum"));


    }*/

}
