import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("NMI.csv", header=None)
K, values = data[0], data[1]

plt.figure('NMI')
plt.plot(K, values)
plt.title('NMI from K-means and categories')
plt.xlabel('K')
plt.ylabel('NMI')
plt.grid(which='both')
plt.tight_layout()
# plt.show()
plt.savefig('NMI.eps', format='eps')
