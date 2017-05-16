#!/bin/bash
clear
./gradlew compileJava
#./gradlew showDepsClasspath
export CP=$(./gradlew showDepsClasspath | grep jar)

java -Dspark.master=local[4]  -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.Doc2Vec dataset/medium-sample.dat.bz2.w2v dataset/medium-sample.dat

#java -Dspark.master=local[4]  -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.Doc2Vec dataset/medium-sample.dat.bz2.w2v dataset/short.dat
