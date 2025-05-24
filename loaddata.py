import numpy as np
import os
from scipy.io import loadmat
import torch
import torch_geometric


def load_final(dataset_name):
	x = loadmat('data/Douban.mat')
	return (x['online_edge_label'][0][1],
			x['online_node_label'],
			x['offline_edge_label'][0][1],
			x['offline_node_label'],
			x['ground_truth'].T,
			x['H'])

def load_dblp(dataset_name):
	f = np.load('data/ACM-DBLP_0.2.npz')
	Afeat, Bfeat = f['x1'].astype('float32'), f['x2'].astype('float32')
	Aedge = f['edge_index1']
	Bedge = f['edge_index2']
	ground_truth = torch.tensor(np.concatenate([f['pos_pairs'], f['test_pairs']], 0).astype('int32')).T

	Afeat = Afeat - np.mean(Afeat, axis=0)
	Bfeat = Bfeat - np.mean(Bfeat, axis=0)

	return Aedge, Afeat, Bedge, Bfeat, ground_truth

def load_AI(dataset_name):
	f = np.load('data/Allmovie_Imdb.npz')
	Afeat, Bfeat = f['feat1'].astype('float32'), f['feat2'].astype('float32')

	Aedge = np.array(np.nonzero(f['edges1']))
	Bedge = np.array(np.nonzero(f['edges2']))
	ground_truth = torch.tensor(f['ground_truth'].astype('int32')).T

	return Aedge, Afeat, Bedge, Bfeat, ground_truth

def load_wd(dataset_name):
	f = np.load('data/MAUIL_douban_weibo.npz')
	Afeat, Bfeat = f['node_feat1'].astype('float32'), f['node_feat2'].astype('float32')

	Afeat = Afeat - Afeat.mean(0)
	Bfeat = Bfeat - Bfeat.mean(0)

	Aedge = f['edge_index1'].T
	Bedge = f['edge_index2'].T
	ground_truth = torch.tensor(f['ground_truth'].astype('int32')).T

	return Aedge, Afeat, Bedge, Bfeat, ground_truth

def load(dataset_name='dblp'):
	if dataset_name in ['dblp']:
		return load_dblp(dataset_name)
	elif dataset_name in ['douban']:
		return load_final(dataset_name)
	elif dataset_name in ['AI']:
		return load_AI(dataset_name)
	elif dataset_name in ['wd']:
		return load_wd(dataset_name)
	else:
		raise ValueError('Not implemented!')