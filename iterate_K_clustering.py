import subprocess
from numpy import logspace

for K in 700 * logspace(0, 6, num=10, base=2):
    print("------> K = ", K)
    subprocess.call(
        "python make.py \
        --class Cluster \
        output/medium-sample.dat.bz2.wpv/ \
        KMeans {} 30".format(int(K)), shell=True)
