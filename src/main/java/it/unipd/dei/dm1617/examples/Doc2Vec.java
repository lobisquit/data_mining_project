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


import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Arrays;

/**
 * Program to transform a wikiPage to a vector
 * @author Dalla Cia Massimo
 */
public class Doc2Vec{

    public static void main(String[] args) {
        String dataPathW2V = args[0];
        String dataPathWiki = args[1];

        //list of stop words
        List<String> stopWords = Arrays.asList("a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", "against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "due", "during", "e", "each", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", "importance", "important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep	keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure	t", "take", "taken", "taking", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "welcome", "we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", "yet", "you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z", "zero");

        //usual setup
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");
        System.out.println("##############################################################################################################################");

        //load wiki pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPathWiki);

        //load word2vec model
        Word2VecModel w2vM = Word2VecModel.load(JavaSparkContext.toSparkContext(sc), dataPathW2V);

        //transform wikipages in a vector
        JavaRDD<Tuple2<Long, Vector>> wikiVectors = pages
                .map((p) -> {
                    String text = p.getText();
                    ArrayList<String> words = Lemmatizer.lemmatize(text);
                    if(p.getCategories().length>0 && words.size()>0) {
                        double norm;
                        Vector v;
                        int cont = 0;
                        ArrayList<String> doc = new ArrayList();

                        //preprocessing of the words
                        for (String word : words) {
                            try {
                                v = w2vM.transform(word);
                                norm = ExsltMath.sqrt(BLAS.dot(v, v));
                                if (!stopWords.contains(word.toLowerCase()) && word.length() > 2 &&
                                        !doc.contains(word.toLowerCase())
                                        ) {
                                    doc.add(word);
                                }
                            } catch (java.lang.IllegalStateException e) {
                            }
                        }

                        //calcuate the vector of the wiki page
                        if(doc.size()==0){
                            Long falseId= new Long(-1);
                            return new Tuple2<>(falseId, Vectors.zeros(100));
                        }
                        Vector q;
                        Vector w = Vectors.zeros(100);//todo rimuovo il valore hardcoded 100
                        for (String word : doc) {
                            q = w2vM.transform(word);
                            BLAS.axpy(1.0, q, w);
                        }
                        double den = (double) doc.size();
                        double scal = (1.0/den);
                        BLAS.scal(scal, w);
                        return new Tuple2<>(p.getId(), w);
                    }else{
                        //case disambigua
                        //todo non voglio tornare un vettore ma non so come fare per ora
                        Long falseId= new Long(-1);
                        return new Tuple2<>(falseId, Vectors.zeros(100));
                    }
                });
        //todo devo salvare in qualche modo la variabile wikiVectors
        //todo così da poterla usare più volte senza ricalcolarla

        //print for debug
        wikiVectors.foreach((tuple)->{
            if(tuple._1()>new Long(0)){
                System.out.println("id "+tuple._1()+"\n"+tuple._2().toJson()+"\n");
            }

        });


        /*
        JavaPairRDD<WikiPage, Vector> pagesAndVectors = pages.zip(tfidf);
        //List<Tuple2<WikiPage, Vector>> firstPages = pagesAndVectors.take(1000);
        Vector v1=w2vM.transform("home");
        Vector v2=w2vM.transform("house");
        double alfa=Distance.cosineDistance(v1,v2);
        */

    }
}