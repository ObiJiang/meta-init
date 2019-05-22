import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
class Generator_fashion_mnist(object):

	def __init__(self,fea=200,k=2):
		self.resize_width = 16
		fashion_mnist = tf.keras.datasets.fashion_mnist
		(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

		# x_train_resized = self._resize(x_train,self.resize_width )
		# x_test_resized = self._resize(x_test,self.resize_width )

		# x_train_f = x_train_resized.reshape(-1, self.resize_width *self.resize_width ).astype(float)/255
		# x_test_f = x_test_resized.reshape(-1, self.resize_width *self.resize_width ).astype(float)/255

		# x_train_f = StandardScaler().fit_transform(x_train_f)
		# x_test_f = StandardScaler().fit_transform(x_test_f)

		x_train_full = x_train.reshape(-1, 28 *28 ).astype(float)/255
		x_test_full = x_test.reshape(-1, 28 *28).astype(float)/255

		x_norm = StandardScaler().fit_transform(x_train_full)
		pca = PCA(n_components=fea, whiten=True)
		x_train_f = pca.fit_transform(x_norm)

		x_norm = StandardScaler().fit_transform(x_test_full)
		pca = PCA(n_components=fea, whiten=True)
		x_test_f = pca.fit_transform(x_norm)

		""" Random Projection based JL lemma """
		# jl_dim = 2
		# ori_dim = 28*28
		# random_matrix = np.random.normal(size=(ori_dim, jl_dim))*np.sqrt(ori_dim/jl_dim)

		# jl_projected_fea = x_train_full @ random_matrix

		# fig, plot = plt.subplots()
		# fig.set_size_inches(50, 50)
		# plt.prism()

		# plot.scatter(jl_projected_fea[:, 0], jl_projected_fea[:, 1], c=y_train)
		# plot.set_xticks(())
		# plot.set_yticks(())

		# plt.tight_layout()
		# plt.savefig("random_projection.png")

		idx_train = [None]*10
		idx_test = [None]*10
		for i in range(10):
			idx_train[i] = np.where(y_train == i)
			idx_test[i] = np.where(y_test == i)

		self.x_train10 = [None]*10
		self.x_test10 = [None]*10
		for i in range(10):
			self.x_train10[i] = x_train_f[idx_train[i][0], :]
			self.x_test10[i] = x_test_f[idx_test[i][0], :]


	def _resize(self, data, width):
		train_data = []
		for img in data:
			resized_img = cv2.resize(img, (width, width))
			train_data.append(resized_img)
		return np.array(train_data)

	def generate(self, size=100, fea=200, k=2, is_train=True, pool_type='FULL'):
		if pool_type == 'FULL':
			pool = [0,1,2,3,4,5,6,7,8,9]
		elif pool_type == 'HARD_TRAIN':
			pool = [0,1,2,3,5,6,7]
		elif pool_type == 'HARD_TEST':
			pool = [4,8,9]
		elif pool_type == 'HALF_TRAIN':
			pool = [0,1,2,3,4]
		elif pool_type == 'HALF_TEST':
			pool = [5,6,7,8,9]
		elif pool_type == 'EASY_TRAIN':
			pool = [0, 1, 2]
		elif pool_type == 'EASY_TEST':
			pool = [0, 1, 2]
		else:
			raise Exception('pool_type not supported')

		np.random.shuffle(pool)
		selected_digits = pool[:k]
		labels = np.arange(size)%k
		selected_digits.sort()
		
		y_list = []
		x_list = []

		for i,digit in enumerate(selected_digits):
			if is_train:
				x = self.x_train10[digit]
			else:
				x = self.x_test10[digit]

			ind = np.random.permutation(len(x))

			num_data_digit = np.sum(labels==i)
			x_list.append(x[ind[:num_data_digit]])
			y_list = y_list + [i]*num_data_digit

		y_all = np.array(y_list)
		x_all = np.concatenate(x_list,axis=0)

		idx = np.arange(size)
		np.random.shuffle(idx)

		# x_all -= np.mean(x_all,axis=0)
		# x_all /= (np.max(np.abs(x_all),axis=0) + 1e-9)
		return x_all[idx], y_all[idx]


if __name__ == '__main__':
	k = 10
	generator = Generator_minst()
	data, labels = generator.generate(1000, 2, k)

	plt.figure()
	for i in range(k):
		plt.scatter(data[np.where(labels == i)[0], 0], data[np.where(labels == i)[0], 1],label=str(i))

	plt.legend()
	plt.show()
