#!/bin/bash
clear
echo "inizio compilazione"
./gradlew compileJava
echo "fine compilazione"
export CP=$(./gradlew showDepsClasspath | grep jar)
#java -Dspark.master=local[4] -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.CountCategories dataset/veryshort.dat
java -Dspark.master=local[4] -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.CountCategories dataset/medium-sample.dat.bz2