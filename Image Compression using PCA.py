
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

img = cv2.imread('still_life_created.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img.shape

plt.imshow(img)

r, g, b = cv2.split(img)

r, g, b = r/255, g/255, b/255

plt.imshow(r)

plt.imshow(g)

plt.imshow(b)

pca_components = 50
pca_r = PCA(n_components=pca_components)
reduced_r = pca_r.fit_transform(r)
pca_g = PCA(n_components=pca_components)
reduced_g = pca_g.fit_transform(g)
pca_b = PCA(n_components=pca_components)
reduced_b = pca_b.fit_transform(b)

combined = np.array([reduced_r.shape, reduced_g.shape, reduced_b.shape])

reconstructed_r = pca_r.inverse_transform(reduced_r)
reconstructed_g = pca_g.inverse_transform(reduced_g)
reconstructed_b = pca_b.inverse_transform(reduced_b)

print(reconstructed_r.shape, reconstructed_g.shape, reconstructed_b.shape)

plt.imshow(reconstructed_r)

reconstructed_img = cv2.merge((reconstructed_r, reconstructed_g, reconstructed_b))
plt.imshow(reconstructed_img)

