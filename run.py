import numpy as np
import tensorflow as tf
import subprocess
import logging
import itertools
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
from tqdm import tqdm

def run_cmd(cmd):
	logging.info("Running command: {:}".format(cmd))
	subprocess.check_call(cmd,shell=True)

run_cmd('rm -rf test_train.txt')
run_cmd('rm -rf test_val.txt')

num_layers_list = [3,5,7]
fea_list = [2,3,4,5]
#fea_list = [25, 50, 100, 150, 200, 250, 300]
num_sequence_list = [1000]
kmeans_k_list = [5]
k_list = [5]

all_comb = itertools.product(num_layers_list, fea_list, num_sequence_list, kmeans_k_list, k_list)
length = len(list(all_comb))
all_comb = itertools.product(num_layers_list, fea_list, num_sequence_list, kmeans_k_list, k_list)
for num_layers,fea,num_sequence,kmeans_k,k in tqdm(all_comb,total=length):
	exp_name = "mnist_numLayers_{:}_fea_{:}_numSeq_{:}_kmeansK_{:}_k_{:}".format(num_layers,fea,num_sequence,kmeans_k,k)
	out_dir = './out'+'/'+exp_name
	summary_dir = './summary_dir' + '/' + exp_name

	desc = "mnist training (0-5) numLayers {:} fea {:} numSeq {:} kmeansK {:} k {:}".format(num_layers,fea,num_sequence,kmeans_k,k)

	# put descriptiont into the output files
	desc_cmd = 'echo "' + desc + '">> test_train.txt'
	run_cmd(desc_cmd)

	desc_cmd = 'echo "' + desc + '">> test_val.txt'
	run_cmd(desc_cmd)

	# # training
	# cmd = 'python model_kmeans_mlp.py --num_layers {:} --fea {:} --num_sequence {:} --kmeans_k {:} --k {:} \
	#  --model_save_dir {:} --summary_dir {:} --mnist_train | tee -a test_train.txt'.format(num_layers, fea, num_sequence, kmeans_k, k, out_dir, summary_dir)
	# run_cmd(cmd)

	# for max_iter in [1,2,3,4,5,10,15,20,25,30]:
	max_iter = 1
	desc_cmd = 'echo "' + "Meta, Meta std, K-means++, K-means++ std, K-means, K-means loss std" + '">> test_val.txt'
	run_cmd(desc_cmd)
	# test
	cmd = 'python model_kmeans_mlp.py --num_layers {:} --fea {:} --num_sequence {:} --kmeans_k {:} --k {:} \
	 --model_save_dir {:} --summary_dir {:} --test --max_iter {:} --tol 0 | tee -a test_val.txt'.format(num_layers, fea, num_sequence, kmeans_k, k, out_dir, summary_dir, max_iter)
	run_cmd(cmd)
