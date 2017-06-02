import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("silhouette.csv", header=None)
K, values = data[0], data[1]

plt.figure('silouette')
plt.plot(K, values)
plt.title('Silhouette score variation')
plt.xlabel('K')
plt.ylabel('Silhouette score')
plt.grid(which='both')
plt.tight_layout()
# plt.show()
plt.savefig('silhouette.eps', format='eps')
