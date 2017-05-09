#!/bin/bash
clear
echo "inizio compilazione"
./gradlew compileJava
echo "fine compilazione"
export CP=$(./gradlew showDepsClasspath | grep jar)
#java -Dspark.master=local[4] -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.CategoriesPreprocessing dataset/veryshort.dat
#java -Dspark.master=local[4] -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.CategoriesPreprocessing dataset/medium-sample.dat.bz2
java -Dspark.master=local[4] -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.CategoriesPreprocessing dataset/veryshort.dat