import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
X, y = make_blobs(n_samples=1000, centers=2, random_state=42, cluster_std=1.5)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Visualize the synthetic dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0]
            [:, 1], color='blue', label='Class 0', s=10)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1]
            [:, 1], color='red', label='Class 1', s=10)
plt.title('Synthetic Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
