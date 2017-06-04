#!/usr/bin/python
import sys
import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster
import pprint

dataFile = open("../output/WikiVector.csv",'r')

dataMatrix = []

mappa = {}
i=0
for line in dataFile:
  d=line.strip().split(',')[0]
  mappa[i]=id
  i+=1
  if numpy.array([float(x) for x in line.strip().split(',')[1:]]).shape[0] == 100:
    dataMatrix.append([float(x) for x in line.strip().split(',')[1:]])
    #print line
    #print numpy.array([float(x) for x in line.strip().split(',')[1:]]).shape

mat = numpy.array(dataMatrix)
print mat.shape
#sys.exit(0)
# clustering
thresh = 0.001
ret = hcluster.fclusterdata(mat, thresh, criterion="distance")


clusters = {}
for (i, item) in enumerate(ret):
  #print mappa[i], item
  idCluster=int(item)
  clusters.setdefault(idCluster,[]).append(mappa[i])



for array in clusters.values():
  print array
