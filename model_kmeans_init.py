import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from utlis.misc import AttrDict, sample_floats, KernelKMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from tensorflow.python.ops.rnn import _transpose_batch_time
from utlis.mnist import Generator_minst
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans,DBSCAN,SpectralClustering
from utlis.mnist import Generator_minst
from sklearn.metrics import normalized_mutual_info_score
import scipy
import math
from utlis.edu import eduGenerate
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from utlis.kmeans import K_means

class MetaCluster():
	def __init__(self,config):
		self.config = config
		self.n_unints = 32
		self.batch_size = config.batch_size
		self.k = config.k # number of clusters
		self.num_sequence = config.num_sequence
		self.memory_size = config.memory_size
		self.fea = config.fea
		self.lr = 0.05
		self.keep_prob = 0.8
		self.alpha = 0.9
		self.knn_k = 5
		self.summary_dir = './summary'
		self.model = self.model()
		self.pic_ind = 1
		self.tol = config.tol
		self.max_iter = config.max_iter
		self.train_ind = 1 
		self.test_ind = 1
		# only keep the core weights
		vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='core')
		vars_ = {var.name.split(":")[0]: var for var in vars}
		self.saver = tf.train.Saver(vars_,max_to_keep=config.max_to_keep)

	def create_dataset(self):
		labels = np.arange(self.num_sequence)%self.k
		np.random.shuffle(labels)

		data = np.zeros((self.num_sequence,self.fea))

		mean = np.random.rand(self.k, self.fea)*5-2.5
		while np.sum(mean[0,:]-mean[1,:]) < self.fea:
			mean = np.random.rand(self.k, self.fea)*5-2.5
		sort_ind = np.argsort(mean[:,0])

		for label_ind,ind in enumerate(sort_ind):
			#cov_factor = np.random.rand(1)*2+1
			cov = np.random.randn(self.fea,self.fea)
			cov = scipy.linalg.orth(cov)
			cov = cov.T @ cov
			data[labels==label_ind,:] = np.random.multivariate_normal(mean[ind, :], cov, (np.sum(labels==label_ind)))

		""" see for yourself """
		if self.config.show_graph:
			for i in range(self.k):
				plt.scatter(twisted_data[labels_new==i,0], twisted_data[labels_new==i,1])
			plt.show()

		centriod = np.expand_dims(self.get_k_means_center(data), axis=0)
		return np.expand_dims(data,axis=0),np.expand_dims(labels,axis=0).astype(np.int32),\
			   centriod
			   #np.expand_dims(mean[sort_ind,:],axis=0)


	def model(self):
		sequences = tf.placeholder(tf.float32, [self.batch_size, None, self.fea])
		centriod = tf.placeholder(tf.float32, [self.batch_size, self.k, self.fea])
		labels = tf.placeholder(tf.int32, [self.batch_size, None])

		# cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_unints,state_is_tuple=True)
		fw_cells = [tf.contrib.rnn.BasicLSTMCell(32) for _ in range(2)]
		fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)

		bw_cells = [tf.contrib.rnn.BasicLSTMCell(32) for _ in range(2)]
		bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)

		""" Define LSTM network """
		with tf.variable_scope('core'):
			# output, states = tf.nn.dynamic_rnn(fw_cell, sequences, dtype=tf.float32)
			output_two, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sequences, dtype=tf.float32)
			output = tf.concat((output_two[0][:,-1,:],output_two[1][:,0,:]), axis=1)
			predicted_centroids = tf.layers.dense(output,self.k*self.fea)

		predicted_centroids_reshape = tf.reshape(predicted_centroids,[-1,self.k,self.fea])

		loss = tf.losses.mean_squared_error(centriod,predicted_centroids_reshape)

		opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

		tf.summary.scalar('loss', loss)

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(self.summary_dir + '/train')
		test_writer = tf.summary.FileWriter(self.summary_dir + '/test')

		return AttrDict(locals())

	def mutual_info(self,true_label,predicted_label):
		if len(true_label.shape) == 1:
			true_label = np.expand_dims(true_label,axis=0)
			predicted_label = np.expand_dims(predicted_label,axis=0)
		nmi_list = []
		for i in range(true_label.shape[0]):
			nmi = normalized_mutual_info_score(true_label[i],predicted_label[i])
			nmi_list.append(nmi)
		return np.mean(nmi_list)

	def er(self,true_label,predicted_label):
		if len(true_label.shape) == 1:
			true_label = np.expand_dims(true_label,axis=0)
			predicted_label = np.expand_dims(predicted_label,axis=0)

		er_list = []
		for i in range(true_label.shape[0]):
			cost = contingency_matrix(true_label[i],predicted_label[i])
			row_ind, col_ind = linear_sum_assignment(-1*cost)
			er = 1-cost[row_ind, col_ind].sum()/self.num_sequence
			er_list.append(er)
		return np.mean(er_list)

	def train(self,data,labels,centriod,sess):
		model = self.model
		summary, _,loss = sess.run([model.merged,model.opt,model.loss],feed_dict={model.sequences:data,model.labels:labels,model.centriod:centriod})
		print("Loss:{}".format(loss))
		model.train_writer.add_summary(summary, self.train_ind)
		self.train_ind += 1
		

	def test(self,data,labels,sess,real_centriods=None,validation=False,centriods=None):
		model = self.model
		if validation:
			summary = sess.run(model.merged,feed_dict={model.sequences:data,model.labels:labels,model.centriod:centriods})
			model.test_writer.add_summary(summary, self.test_ind)
			self.test_ind += 1
		centriods = sess.run(model.predicted_centroids_reshape,feed_dict={model.sequences:data,model.labels:labels})
		for data_one,label,centriod in zip(data,labels,centriods):
			kmeans = KMeans(n_clusters=self.k, init=centriod,tol=self.tol, max_iter=self.max_iter, algorithm='full').fit(data_one)
			# kmeans = K_means(k=self.k,tolerance=self.tol,max_iterations=self.max_iter)
			# kmeans_labels = kmeans.clustering(data_one,init=centriod)
			kmeans_labels = kmeans.labels_
			meta_plus_kmeans_nmi = self.mutual_info(label,kmeans_labels)
			meta_plus_kmeans_er = self.er(label,kmeans_labels)
			# print("Kmeans:{} and Meta-Kmeans: {}".format(kmeans_nmi,meta_plus_kmeans_nmi))
			# print(centriod)
			# print(real_centriod)
		return meta_plus_kmeans_nmi,meta_plus_kmeans_er

	def save_model(self, sess, epoch):
		print('\nsaving model...')

		# create path if not around
		model_save_path = self.config.model_save_dir
		if not os.path.isdir(model_save_path):
			os.makedirs(model_save_path)

		model_name = '{}/model'.format(model_save_path)
		save_path = self.saver.save(sess, model_name, global_step = epoch)
		print('model saved at', save_path, '\n\n')

	def get_k_means_center(self,data):
		kmeans = KMeans(n_clusters=self.k,random_state=0,algorithm='full').fit(data)
		# kmeans = K_means(k=self.k,tolerance=self.tol,max_iterations=self.max_iter)
		# kmeans_labels = kmeans.clustering(data)
		# cluster_centers = kmeans.centroids
		cluster_centers = kmeans.cluster_centers_
		sort_ind = np.argsort(np.sum(cluster_centers[:,:],axis=1))

		return cluster_centers[sort_ind,:]


if __name__ == '__main__':
	# arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--test', default=False, action='store_true')
	parser.add_argument('--mnist_train', default=False, action='store_true')
	parser.add_argument('--show_graph', default=False, action='store_true')
	parser.add_argument('--show_comparison_graph', default=False, action='store_true')
	parser.add_argument('--max_to_keep', default=3, type=int)
	parser.add_argument('--model_save_dir', default='./out')
	parser.add_argument('--batch_size', default=100, type=int)
	parser.add_argument('--fea', default=2, type=int)
	parser.add_argument('--num_sequence', default=100, type=int)
	parser.add_argument('--memory_size', default=100, type=int)
	parser.add_argument('--training_exp_num', default=50, type=int)
	parser.add_argument('--k',default=2,type=int)
	parser.add_argument('--tol',default=1e-4,type=float)
	parser.add_argument('--max_iter',default=300,type=int)
	parser.add_argument('--algo',default='auto')

	config = parser.parse_args()
	generator = Generator_minst(fea=config.fea)

	if not config.test:
		metaCluster = MetaCluster(config)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# training
			for train_ind in tqdm(range(int(config.training_exp_num))):
				data_list = []
				labels_list = []
				centriod_list = []
				for _ in range(config.batch_size):
					if config.mnist_train:
						data_one, labels_one = generator.generate(metaCluster.num_sequence, metaCluster.fea)
						centriod_one = np.expand_dims(metaCluster.get_k_means_center(data_one), axis=0)
						data_one = np.expand_dims(data_one, axis=0)
						labels_one = np.expand_dims(labels_one, axis=0)
					else:
						data_one, labels_one, centriod_one = metaCluster.create_dataset()
					data_list.append(data_one)
					labels_list.append(labels_one)
					centriod_list.append(centriod_one)
				data = np.concatenate(data_list)
				labels = np.concatenate(labels_list)
				centriods = np.concatenate(centriod_list)
				metaCluster.train(data,labels,centriods,sess)

				if train_ind % 10 == 0:
					print('-----validation-----')
					# validation
					data_list = []
					labels_list = []
					centriod_list = []
					for _ in range(config.batch_size):
						if config.mnist_train:
							data_one, labels_one = generator.generate(metaCluster.num_sequence, metaCluster.fea)
							centriod_one = np.expand_dims(metaCluster.get_k_means_center(data_one), axis=0)
							data_one = np.expand_dims(data_one, axis=0)
							labels_one = np.expand_dims(labels_one, axis=0)
						else:
							data_one, labels_one, centriod_one = metaCluster.create_dataset()
						data_list.append(data_one)
						labels_list.append(labels_one)
						centriod_list.append(centriod_one)
					data = np.concatenate(data_list)
					labels = np.concatenate(labels_list)
					centriods = np.concatenate(centriod_list)
					metaCluster.test(data,labels,sess,validation=True,centriods=centriods)
			# saving models ...
			metaCluster.save_model(sess,config.training_exp_num)

	else:
		config.batch_size = 1
		metaCluster = MetaCluster(config)
		with tf.Session() as sess:
			""" Reload parameters """
			sess.run(tf.global_variables_initializer())
			vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='core')
			vars_ = {var.name.split(":")[0]: var for var in vars}
			saver = tf.train.Saver(vars_, max_to_keep=config.max_to_keep)
			save_dir = config.model_save_dir

			checkpoint = tf.train.get_checkpoint_state(save_dir)
			assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
			save_path = checkpoint.model_checkpoint_path
			print("Loading saved model from {}".format(save_path))
			saver.restore(sess, save_path)

			kmeans_list_er = []
			meta_plus_kmeans_list_er = []

			kmeans_list_nmi = []
			meta_plus_kmeans_list_nmi = []
			for _ in tqdm(range(100)):
				# data, labels, centriods = metaCluster.create_dataset()
				data, labels = generator.generate(metaCluster.num_sequence, metaCluster.fea, is_train=False)
				data = np.squeeze(data)
				labels = np.squeeze(labels)

				""" Kmeans """
				kmeans = KMeans(n_clusters=metaCluster.k,random_state=0,tol=metaCluster.tol, max_iter=metaCluster.max_iter, algorithm='full').fit(data)
				# kmeans = K_means(k=metaCluster.k,tolerance=metaCluster.tol,max_iterations=metaCluster.max_iter)
				# kmeans_labels = kmeans.clustering(data)
				kmeans_labels = kmeans.labels_
				kmeans_nmi = metaCluster.mutual_info(labels,kmeans_labels)
				kmeans_er = metaCluster.er(labels,kmeans_labels)
				kmeans_list_er.append(kmeans_er)
				kmeans_list_nmi.append(kmeans_nmi)


				data = np.expand_dims(data, axis=0)
				labels = np.expand_dims(labels, axis=0)
				meta_plus_kmeans_nmi,meta_plus_kmeans_er = metaCluster.test(data,labels,sess)
				meta_plus_kmeans_list_er.append(meta_plus_kmeans_er)
				meta_plus_kmeans_list_nmi.append(meta_plus_kmeans_nmi)
				
			print("Error Rate: Kmeans:{:.2f}+-{:.2f} and Meta-Kmeans: {:.2f}+-{:.2f}".format(np.mean(kmeans_list_er),np.std(kmeans_list_er),np.mean(meta_plus_kmeans_list_er),np.std(meta_plus_kmeans_list_er)))
			print("NMI: Kmeans:{:.2f}+-{:.2f} and Meta-Kmeans: {:.2f}+-{:.2f}".format(np.mean(kmeans_list_nmi),np.std(kmeans_list_nmi),np.mean(meta_plus_kmeans_list_nmi),np.std(meta_plus_kmeans_list_nmi)))

