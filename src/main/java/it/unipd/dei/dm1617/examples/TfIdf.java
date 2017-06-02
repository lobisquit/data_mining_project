package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Convert documents in "bag of words" representations using TfIdf
 */
public class TfIdf {

    private static List<String> readFile(String path) {
        Scanner s = null;
        try {
            s = new Scanner(new File(path));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        List<String> lines = new ArrayList<>();
        try {
            while (s.hasNext()){
                lines.add(s.next());
            }
        } catch (NullPointerException e) {
            e.printStackTrace();
        }
        s.close();
        return lines;
    }

    private static void toFile (String filename, String[] x) throws IOException{
        BufferedWriter outputWriter;
        outputWriter = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < x.length; i++) {
            outputWriter.write(x[i]);
            outputWriter.newLine();
        }
        outputWriter.flush();
        outputWriter.close();
    }

    /**
     * @param args
     *  args[0] : path of the dataset
     *  args[1] : number of lemmas considered for every page
     */
    public static void main(String[] args) {
        String dataPath = args[0];
        int vocabularySize = Integer.parseInt(args[1]);

        // Load list of stop words
        List<String> stopWords = readFile("dataset/stop_words.txt");

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

        // Remove stopwords and short lemmas
        JavaRDD<ArrayList<String>> purgedLemmas = lemmas.map(array -> {
            ArrayList<String> newArray = new ArrayList<>();
            for (String word : array) {
                // remove words that are too short or that are stop words
                if (!stopWords.contains(word.toLowerCase()) && word.length() > 2) {
                    newArray.add(word);
                }
            }
            return newArray;
        });

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
        CountVectorizer cv = new CountVectorizer();
        JavaRDD<Vector> tf = cv
                .setVocabularySize(vocabularySize)
                .transform(purgedLemmas)
                .cache();

        String[] vocabulary = cv.getVocabulary();

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
        // From PairRDD to RDD
        JavaRDD<Tuple2<WikiPage, Vector>> pavRDD = JavaPairRDD.toRDD(pagesAndVectors).toJavaRDD();

        try {
            pavRDD.saveAsObjectFile("output/TfIdf--"+dataPath.split("/")[1]+"--vocabulary:"+vocabularySize+".tfidf");
            toFile("output/TfIdfVocabulary--"+dataPath.split("/")[1]+"--vocabulary:"+vocabularySize+".txt", vocabulary);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

