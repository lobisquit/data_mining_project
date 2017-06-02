import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("KMeans.csv", header=None)
K, values = data[0], data[1]

plt.figure('Kmeans obj')
plt.plot(K, values)
plt.title('Objective function variation')
plt.xlabel('K')
plt.ylabel('K-means objective function')
plt.grid(which='both')
plt.tight_layout()
# plt.show()
plt.savefig('NMI-kmeans-categories.eps', format='eps')

plt.figure('Kmeans obj derivative')
derivative = []
for i in range(1, len(K)):
    d = (values[i-1] - values[i]) / (K[i-1] - K[i])
    derivative.append(d)

plt.semilogx(K[1:], derivative)
plt.title('Objective function variation')
plt.xlabel('K')
plt.ylabel('K-means derivative')
plt.grid(which='major')
plt.tight_layout()
# plt.show()
plt.savefig('KMeansDerivative.eps', format='eps')
