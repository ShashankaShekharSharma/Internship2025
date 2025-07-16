import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Sample 2D data
np.random.seed(42)
X = np.random.multivariate_normal(mean=[0, 0], cov=[[3, 2], [2, 2]], size=10)

# 2. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA transformation
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
X_projected_back = pca.inverse_transform(X_pca)

# 4. Get PCA components and mean
pc_vector = pca.components_[0]   # Direction vector
mean_point = np.mean(X_scaled, axis=0)  # Center of the data

# 5. Generate line coordinates for PCA axis
line_length = 5
line_coords = np.vstack([mean_point + t * pc_vector for t in np.linspace(-line_length, line_length, 100)])

# 6. Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], color='blue', label='Original Data')
plt.scatter(X_projected_back[:, 0], X_projected_back[:, 1], color='red', label='Projected Points')

# Lines from each point to its projection
for i in range(len(X_scaled)):
    plt.plot([X_scaled[i, 0], X_projected_back[i, 0]], 
             [X_scaled[i, 1], X_projected_back[i, 1]], 
             'gray', linestyle='--', linewidth=0.7)

# Plot the principal component line
plt.plot(line_coords[:, 0], line_coords[:, 1], color='green', linestyle='-', linewidth=2, label='PCA Line')

plt.title("PCA Projection and Principal Component Axis")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
