import subprocess
from numpy import logspace

for K in 700 * logspace(0, 6, num=10, base=2) * 2**0.5:
    print("------> K = ", K)
    subprocess.call(
        "python3 make.py \
        --class Cluster --master local[4] \
        output/medium-sample.dat.wpv/ \
        KMeans {} 30".format(int(K)), shell=True)
