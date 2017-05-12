package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.commons.collections.iterators.ArrayListIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.xalan.lib.ExsltMath;
import scala.Tuple2;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.feature.IDFModel;


import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Program to transform a document to a vector
 */
public class Doc2Vec{

    public static void main(String[] args) {
        String dataPathW2V = args[0];
        String dataPathWiki = args[1];

        //usual setup
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");
        System.out.println("##############################################################################################################################");

        //load word to vec model
        Word2VecModel w2vM=Word2VecModel.load(JavaSparkContext.toSparkContext(sc),dataPathW2V);



        //TFIDF
        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPathWiki);
        JavaRDD<String> texts = pages.map((p) -> p.getText());
        JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();
        //System.out.println(lemmas.first());

        JavaRDD<Vector> tf = new CountVectorizer().setVocabularySize(100).transform(lemmas).cache();

        IDF idf=new IDF();
        IDFModel idfM = idf.fit(tf);
        JavaRDD<Vector> tfidf = idfM.transform(tf);
        System.out.println(tfidf.first());




        /*
        JavaPairRDD<WikiPage, Vector> pagesAndVectors = pages.zip(tfidf);

        System.out.println("id: "+pagesAndVectors.first()._1.getId());
        System.out.println(pagesAndVectors.first()._2.toJson());


        List<Tuple2<WikiPage, Vector>> firstPages = pagesAndVectors.take(1000);
        */





        /*
        Vector v1=w2vM.transform("home");
        Vector v2=w2vM.transform("house");

        double alfa=Distance.cosineDistance(v1,v2);
        System.out.println("alfa "+alfa);
        */
    }


    /*public Vector text2Vec(String text){


    }*/



}