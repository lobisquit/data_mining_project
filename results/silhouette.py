import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("simpleSilhouette.csv", header=None)
K, values = data[0], data[1]

plt.figure('silhouette')
plt.plot(K, values)
plt.title('Simplified Silhouette Coefficient variation')
plt.xlabel('K')
plt.ylabel('Simplified Silhouette Coefficient')
plt.grid(which='both')
plt.tight_layout()
# plt.show()
plt.savefig('simplifiedSilhouette.eps', format='eps')


plt.figure('Simplified Silhouette derivative')
derivative = []
for i in range(1, len(K)):
    d = (values[i-1] - values[i]) / (K[i-1] - K[i])
    derivative.append(d)

plt.semilogx(K[1:], derivative)
plt.title('Simplified Silhouette function variation')
plt.xlabel('K')
plt.ylabel('Simplified Silhouette derivative')
plt.grid(which='major')
plt.tight_layout()
# plt.show()
plt.savefig('simplifiedSilhouetteDerivative.eps', format='eps')
