import numpy as np
import matplotlib.pyplot as plt

# 1. Generate sample 2D data
np.random.seed(42)
X = np.random.multivariate_normal(mean=[0, 0], cov=[[3, 2], [2, 2]], size=10)

# 2. Standardize the data manually
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# 3. Compute the covariance matrix
cov_matrix = np.cov(X_scaled.T)

# 4. Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 5. Sort eigenvectors by descending eigenvalues
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 6. Select the top 1 principal component
pc1 = eigenvectors[:, 0]

# 7. Project data onto the first principal component
X_pca = X_scaled @ pc1.reshape(-1, 1)  # shape (10, 1)
X_projected = X_pca @ pc1.reshape(1, -1)  # Projected back in 2D space

# 8. Plotting
plt.figure(figsize=(8, 6))

# Original data
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], label='Original Data', color='blue')

# Projected points
plt.scatter(X_projected[:, 0], X_projected[:, 1], label='Projected Points', color='red')

# Draw lines from original to projected
for i in range(X_scaled.shape[0]):
    plt.plot([X_scaled[i, 0], X_projected[i, 0]],
             [X_scaled[i, 1], X_projected[i, 1]],
             color='gray', linestyle='--', linewidth=0.7)

# Draw the principal component line
line_length = 5
mean_point = np.mean(X_scaled, axis=0)
line_points = np.array([
    mean_point - line_length * pc1,
    mean_point + line_length * pc1
])
plt.plot(line_points[:, 0], line_points[:, 1], color='green', label='PCA Line', linewidth=2)

plt.title("PCA from Scratch: Projection and Principal Axis")
plt.xlabel("Feature 1 (standardized)")
plt.ylabel("Feature 2 (standardized)")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
