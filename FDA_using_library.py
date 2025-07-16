import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# 1. Generate synthetic 2D data for 2 classes
np.random.seed(42)
class1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], 10)
class2 = np.random.multivariate_normal([3, 3], [[1, 0.75], [0.75, 1]], 10)

X = np.vstack((class1, class2))
y = np.array([0]*10 + [1]*10)

# 2. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply Linear Discriminant Analysis (FDA)
lda = LinearDiscriminantAnalysis(n_components=1)
X_fda = lda.fit_transform(X_scaled, y)

# 4. Project the data back onto the 2D plane
# We project the 1D points back using the LDA direction vector
w = lda.coef_[0]
w = w / np.linalg.norm(w)  # normalize the direction vector
X_projected = X_fda @ w.reshape(1, -1)

# 5. Plotting
plt.figure(figsize=(8, 6))

# Plot original data by class
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='green', label='Class 1')

# Plot projected points
plt.scatter(X_projected[:, 0], X_projected[:, 1], color='red', label='Projected Points (1D)')

# Draw projection lines
for i in range(X_scaled.shape[0]):
    plt.plot([X_scaled[i, 0], X_projected[i, 0]],
             [X_scaled[i, 1], X_projected[i, 1]],
             'gray', linestyle='--', linewidth=0.7)

# Draw the FDA line
mean_point = np.mean(X_scaled, axis=0)
line_length = 5
line_coords = np.vstack([mean_point + t * w for t in np.linspace(-line_length, line_length, 100)])
plt.plot(line_coords[:, 0], line_coords[:, 1], color='black', label='FDA Line', linewidth=2)

plt.title("Fisher Discriminant Analysis (FDA) using scikit-learn")
plt.xlabel("Feature 1 (standardized)")
plt.ylabel("Feature 2 (standardized)")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
