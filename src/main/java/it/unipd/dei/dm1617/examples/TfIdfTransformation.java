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
public class TfIdfTransformation {

  public static void main(String[] args) {
    String dataPath = args[0];

    // Usual setup
    SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load dataset of pages
    JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

    // Get text out of pages
    JavaRDD<String> texts = pages.map((p) -> p.getText());

    // Get the lemmas. It's better to cache this RDD since the
    // following operation, lemmatization, will go through it two
    // times.
    JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

    // Transform the sequence of lemmas in vectors of counts in a
    // space of 100 dimensions, using the 100 top lemmas as the vocabulary.
    // This invocation follows a common pattern used in Spark components:
    //
    //  - Build an instance of a configurable object, in this case CountVectorizer.
    //  - Set the parameters of the algorithm implemented by the object
    //  - Invoke the `transform` method on the configured object, yielding
    //  - the transformed dataset.
    //
    // In this case we also cache the dataset because the next step,
    // IDF, will perform two passes over it.
    JavaRDD<Vector> tf = new CountVectorizer().setVocabularySize(100).transform(lemmas).cache();

    // Same as above, here we follow the same pattern, with a small
    // addition. Some of these "configurable" objects configure their
    // internal state by means of an invocation of their `fit` method
    // on a dataset. In this case, the Inverse Document Frequence
    // algorithm needs to know about the term frequencies across the
    // entire input dataset before rescaling the counts of the single
    // vectors, and this is what happens inside the `fit` method invocation.
    JavaRDD<Vector> tfidf = new IDF()
      .fit(tf)
      .transform(tf);

    // In this last step we "zip" toghether the original pages and
    // their corresponding tfidf vectors. We can perform this
    // operation safely because we did no operation changing the order
    // of pages and vectors within their respective datasets,
    // therefore the first vector corresponds to the first page and so
    // on.
    JavaPairRDD<WikiPage, Vector> pagesAndVectors = pages.zip(tfidf);
    List<Tuple2<WikiPage, Vector>> firstPages = pagesAndVectors.take(1000);

    //my adds-----------------------------------------------------------------------
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


    System.out.println("################### START ############################\n");
    //Vector parigi=w2vM.transform("parigi");
    //Vector roma=w2vM.transform("roma");

    //System.out.println(parigi.toJson());

    System.out.println("\n################### END ############################");
    //-----------------------------------------------------------------------
  }

  public static void printCategories(List<Tuple2<WikiPage, Vector>> firstPages){
    for(int i=0; i<firstPages.size()-1; i++){
      //print categories of a page
      //example of a page with no category
      //if(firstPages.get(i)._1().getTitle().equals("Conductor")){
        System.out.println(firstPages.get(i)._1().getTitle()+"\n");
        System.out.println("lunghezza del testo "+firstPages.get(i)._1().getText().length()+"\n");
        for(String str:firstPages.get(i)._1().getCategories()){
          System.out.println("\t--> "+str);
        }
        System.out.println("--------");
      //}
    }
  }

  public static void print(List<Tuple2<WikiPage, Vector>> firstPages){
    double dist;

    for(int i=0; i<firstPages.size()-1; i++){
      if(firstPages.get(i)._1().getCategories().length!=0){
        for(int j=1; j<firstPages.size(); j++){
          if(firstPages.get(j)._1().getCategories().length!=0){
            dist = Distance.cosineDistance(firstPages.get(i)._2(), firstPages.get(j)._2());
            if(dist>=0.99){
              printDistance(0,i,dist,firstPages.get(i)._1().getTitle(), firstPages.get(j)._1().getTitle());
            }
          }
        }
      }
    }
  }

  public static void printDistance(int doc1, int doc2, double dist, String tit1, String tit2){
    System.out.println("Cosine distance between "+tit1+" and "+tit2+" = "+dist);
  }

}
