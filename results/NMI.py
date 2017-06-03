import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("NMI-KMeans-categories.csv")
K, overlapping, ranked = data["K"], data["Overlapping"], data["Ranked"]

plt.figure('NMI')
plt.plot(K, overlapping)
plt.title('NMI between K-means and overlapping categories')
plt.xlabel('K')
plt.ylabel('NMI')
plt.grid(which='both')
plt.tight_layout()
# plt.show()
plt.savefig('../latex/Figures/NMI-kmeans-overlapping-categories.eps', format='eps')

plt.figure('NMI2')
plt.plot(K, ranked)
plt.title('NMI between K-means and ranked categories')
plt.xlabel('K')
plt.ylabel('NMI')
plt.grid(which='both')
plt.tight_layout()
# plt.show()
plt.savefig('../latex/Figures/NMI-kmeans-ranked-categories.eps', format='eps')
