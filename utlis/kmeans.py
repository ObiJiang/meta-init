import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
class K_means():
	def __init__(self, k =3, tolerance = 0.0001, max_iterations = 500):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations

	def dist(self, centroid, points):
		return np.linalg.norm(points - centroid, axis=1)

	def initialize_clusters(self, points, k):
		"""Initializes clusters as k randomly selected points from points."""
		return points[np.random.randint(points.shape[0], size=k)]

	def clustering(self,points,init=None):
		if init is None:
			self.centroids = self.initialize_clusters(points, self.k)		
		else:
			self.centroids = init
		original_centroid = self.centroids
		# Loop for the maximum number of iterations
		classes = np.zeros(points.shape[0], dtype=np.float64)
		distances = np.zeros([points.shape[0], self.k], dtype=np.float64)
		for i in range(self.max_iterations):
			
			# Assign all points to the nearest centroid
			for i, c in enumerate(self.centroids):
				distances[:, i] = self.dist(c, points)
				
			# Determine class membership of each point
			# by picking the closest centroid
			classes = np.argmin(distances, axis=1)
			
			# Update centroid location using the newly
			# assigned data point classes
			for c in range(self.k):
				points_in_class = points[classes == c]
				if points_in_class.shape[0] == 0:
					self.centroids[c] = np.zeros(points.shape[1])
				else:
					self.centroids[c] = np.mean(points_in_class, 0)

			if np.sum((self.centroids - original_centroid)/(original_centroid+1e-10)*100) < self.tolerance:
				return classes

			original_centroid = self.centroids

		return classes


if __name__ == '__main__':
	X, y = make_blobs(centers=3, n_samples=500, random_state=1)
	kmeans = K_means()
	labels = kmeans.clustering(X)

	group_colors = ['skyblue', 'coral', 'lightgreen']
	colors = [group_colors[j] for j in labels]
	fig, ax = plt.subplots(figsize=(4,4))
	ax.scatter(X[:,0], X[:,1], color=colors, alpha=0.5)
	ax.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], color=['blue', 'darkred', 'green'], marker='o', lw=2)
	ax.set_xlabel('$x_0$')
	ax.set_ylabel('$x_1$');
	plt.show()