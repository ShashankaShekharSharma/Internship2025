import numpy as np
import matplotlib.pyplot as plt

# 1. Create 2-class 2D sample data
np.random.seed(42)
class1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], 10)
class2 = np.random.multivariate_normal([3, 3], [[1, 0.75], [0.75, 1]], 10)

X = np.vstack((class1, class2))
y = np.array([0]*10 + [1]*10)  # Class labels: 0 and 1

# 2. Standardize the data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# 3. Separate the classes
X1 = X_scaled[y == 0]
X2 = X_scaled[y == 1]

# 4. Compute class means
mean1 = np.mean(X1, axis=0)
mean2 = np.mean(X2, axis=0)

# 5. Compute within-class scatter matrix Sw
Sw = np.cov(X1, rowvar=False) + np.cov(X2, rowvar=False)

# 6. Compute between-class scatter vector Sb
mean_diff = (mean1 - mean2).reshape(-1, 1)
Sb = mean_diff @ mean_diff.T

# 7. Compute the FDA direction: Solve Sw^-1 * (mean1 - mean2)
w = np.linalg.inv(Sw) @ (mean1 - mean2)

# 8. Normalize the direction vector
w = w / np.linalg.norm(w)

# 9. Project all points onto w
X_fda = X_scaled @ w.reshape(-1, 1)
X_projected = X_fda @ w.reshape(1, -1)

# 10. Plot
plt.figure(figsize=(8, 6))

# Plot original data
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='green', label='Class 1')

# Plot projected points
plt.scatter(X_projected[:, 0], X_projected[:, 1], color='red', label='Projected Points')

# Connect original to projected points
for i in range(X_scaled.shape[0]):
    plt.plot([X_scaled[i, 0], X_projected[i, 0]], 
             [X_scaled[i, 1], X_projected[i, 1]], 
             'gray', linestyle='--', linewidth=0.7)

# Plot the FDA line
mean_point = np.mean(X_scaled, axis=0)
line_length = 5
line_coords = np.vstack([mean_point + t * w for t in np.linspace(-line_length, line_length, 100)])
plt.plot(line_coords[:, 0], line_coords[:, 1], color='black', label='FDA Line', linewidth=2)

plt.title("Fisher Discriminant Analysis (FDA) Projection")
plt.xlabel("Feature 1 (standardized)")
plt.ylabel("Feature 2 (standardized)")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
