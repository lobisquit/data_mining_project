package it.unipd.dei.dm1617;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;

import java.io.Serializable;
import java.util.Arrays;


public class WikiV implements Serializable {
    private Integer cluster;
    private Long wikiId;
    private Vector v;

    public static Encoder<WikiV> getEncoder() {
      return Encoders.bean(WikiV.class);
    }

    public void setCluster(Integer cluster2){
      cluster=cluster2;
    }
    public void setWikiId(Long index){
      wikiId=index;
    }
    public void setVector(Vector vet){
      v=vet;
    }
    public Integer getCluster(){
      return cluster;
    }
    public Long getWikiId(){
      return wikiId;
    }
    public Vector getVector(){
      return v;
    }
    public String toString(){
        return cluster+","+wikiId+","+v.toJson();
    }
  
}
