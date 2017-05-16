#!/bin/bash
clear
./gradlew compileJava
#./gradlew showDepsClasspath
export CP=$(./gradlew showDepsClasspath | grep jar)

java -Dspark.master=local[4]  -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.Cluster output/medium-sample.dat.wv/part-00000
