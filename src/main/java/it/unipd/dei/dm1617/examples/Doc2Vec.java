package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.commons.collections.iterators.ArrayListIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.xalan.lib.ExsltMath;
import scala.Float;
import scala.Tuple2;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.feature.HashingTF;


import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

/**
 * Program to transform a wikiPage to a vector, the result will be saved in a directory
 * with the same name given in args[1] with the add of .wv in the end
 * @author Dalla Cia Massimo
 */
public class Doc2Vec{
    public static List<String> readFile(String path) {
        Scanner s = null;
        try {
            s = new Scanner(new File(path));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        List<String> lines = new ArrayList<String>();
        while (s.hasNext()){
            lines.add(s.next());
        }
        s.close();
        return lines;
    }

    public static void main(String[] args) {
        // input Word2Vec model path
        String dataPathW2V = args[0];

        // input wikipedia corpus path
        String dataPathWiki = args[1];

        // load list of stop words
        List<String> stopWords = readFile("dataset/stop_words.txt");

        // usual Spark setup
        SparkConf conf = new SparkConf(true).setAppName("Doc2Vec");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // mark the starting point of our subsequent messages
        System.out.println("###################################################" +
            "#################################################################");

        // load wiki pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPathWiki);

        // load word2vec model
        Word2VecModel w2vM = Word2VecModel.load(
            JavaSparkContext.toSparkContext(sc), dataPathW2V);

        // transform wikipages in a vector
        JavaRDD<Tuple2<Long, Vector>> wikiVectors = pages
                .map((p) -> {
                    String text = p.getText();
                    ArrayList<String> words = Lemmatizer.lemmatize(text);
                    // don't perform analysis on articles with no categories
                    // (such as disambiguation) and with no text
                    if(p.getCategories().length>0 && words.size()>0) {
                        // save all words of the article in one list
                        ArrayList<String> doc = new ArrayList();
                        for (String word : words) {
                            // remove words that are too short, that are stop words
                            // and the ones that are already in the bag
                            if (!stopWords.contains(word.toLowerCase()) &&
                                    word.length() > 2 &&
                                    !doc.contains(word.toLowerCase())) {
                                doc.add(word);
                            }
                        }

                        // compute the vector of the wiki page
                        if(doc.size()==0) {
                            // case in which the preprocessing deleted all words
                            // here a fake id is employed to filter out these rows in
                            // the reduce phase
                            Long fakeId= new Long(-1);
                            return new Tuple2<>(fakeId, Vectors.zeros(100));
                        }

                        Vector q;
                        Vector w = Vectors.zeros(100); //todo rimuovo il valore hardcoded 100
                        for (String word : doc) {
                            try {
                                q = w2vM.transform(word);
                                BLAS.axpy(1.0, q, w);
                            } catch (java.lang.IllegalStateException e) {}
                        }
                        double den = (double) doc.size();
                        double scal = (1.0/den);
                        BLAS.scal(scal, w);
                        return new Tuple2<>(p.getId(), w);
                    } else {
                        // here a fake id is employed to filter out these rows in
                        // the reduce phase
                        Long falseId= new Long(-1);
                        return new Tuple2<>(falseId, Vectors.zeros(100));
                    }
                });

        // save the wikiVectors javaRDD with the name of the input WikiPages RDD
        String[] parts = dataPathWiki.split("/");
        wikiVectors.saveAsObjectFile("output/" + parts[1] + ".wv");

        //print for debug if needed
        /*wikiVectors.foreach((tuple)->{
            if(tuple._1()>new Long(0)){
                System.out.println("id "+tuple._1()+"\n"+tuple._2().toJson()+"\n");
            }
        });*/
    }
}
