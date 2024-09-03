import pickle
import torch
import vedo
import trimesh
import scipy
import numpy as np
import networkx as nx 
import seaborn as sns
from tqdm import tqdm
from smpl import SMPL
import tensorflow as tf
import polyscope as ps
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.sparse.csgraph as csgraph



class Visualizer:

	# visualizing the correspondences based on HKS Values, but on meshes
	def plot_hks_corresp(k, f_dim, eval_smpl, evec_smpl, eval_rabit, evec_rabit, j_smpl, j_rabit):
	    print("----- Calculating the HKS on the Mesh ---------")

	    hks_smpl_mesh = compute_hks(eval_smpl[:k], evec_smpl[:,:k], f_dim) # (B,V,feature_dim)
	    hks_rabit_mesh = compute_hks(eval_rabit[:k], evec_rabit[:,:k], f_dim) # (B,V,feature_dim)
	    
	    color_smpl = hks_smpl_mesh[0].numpy()[:,:3]
	    color_smpl = (color_smpl - color_smpl.min())/(color_smpl.max()-color_smpl.min())
	    color_rabit = hks_rabit_mesh[0].numpy()[:,:3]
	    color_rabit = (color_rabit - color_rabit.min())/(color_rabit.max()-color_rabit.min())

	    # trimesh.Trimesh(joints,vertex_colors = color_smpl*255).export('smpl.ply')
	    # trimesh.Trimesh(rabit_joints,vertex_colors = color_rabit*255).export('rabit.ply')

	    color_smpl = color_smpl.astype(np.float64)
	    color_rabit = color_rabit.astype(np.float64)

	    ps_smpl = ps.register_point_cloud("SMPL", j_smpl, enabled=True, radius=0.01, material='flat')
	    ps_smpl.add_color_quantity("color of the joints", color_smpl)

	    ps_rabit = ps.register_point_cloud("RaBit", ((j_rabit + [0.6, 0.0, 0.0])* 4.0), enabled=True, radius=0.01, material='flat')
	    ps_rabit.add_color_quantity("color of the joints", color_rabit)

	    ps.show()

	# visualizing the correspondences based on HKS Values
	def visualize_corresp(k, f_dim, eval_smpl, evec_smpl, eval_rabit, evec_rabit, j_smpl, j_rabit):
	    print("----- Calculating the HKS on the Mesh ---------")

	    hks_smpl_mesh = compute_hks(eval_smpl[:k], evec_smpl[:,:k], f_dim) # (B,V,feature_dim)
	    hks_rabit_mesh = compute_hks(eval_rabit[:k], evec_rabit[:,:k], f_dim) # (B,V,feature_dim)
	    
	    color_smpl = hks_smpl_mesh[0].numpy()[:,:3]
	    color_smpl = (color_smpl - color_smpl.min())/(color_smpl.max()-color_smpl.min())
	    color_rabit = hks_rabit_mesh[0].numpy()[:,:3]
	    color_rabit = (color_rabit - color_rabit.min())/(color_rabit.max()-color_rabit.min())

	    # trimesh.Trimesh(joints,vertex_colors = color_smpl*255).export('smpl.ply')
	    # trimesh.Trimesh(rabit_joints,vertex_colors = color_rabit*255).export('rabit.ply')

	    print(f"Dimension of the color: {color_smpl.shape}")
	    color_smpl = color_smpl.astype(np.float64)
	    color_rabit = color_rabit.astype(np.float64)

	    ps_smpl = ps.register_point_cloud("SMPL", j_smpl, enabled=True, radius=0.01, material='flat')
	    ps_smpl.add_color_quantity("color of the joints", color_smpl)

	    ps_rabit = ps.register_point_cloud("RaBit", ((j_rabit + [0.6, 0.0, 0.0])* 4.0), enabled=True, radius=0.01, material='flat')
	    ps_rabit.add_color_quantity("color of the joints", color_rabit)

	    ps.show()


	# heat map for the Adjacency Matrix
	def heat_map_adj_mat(matrix):
		sns.heatmap(matrix, annot=True)
		plt.show()


	# Plotting correspondences accross SMPL and RaBit
	def plot_corresp(corresp, smpl_joints, rabit_joints, smpl_edges, rabit_edges, scale, trans):


	    plt.figure(figsize=(24, 24))

	    # plotting for smpl
	    plt.scatter(smpl_joints[:, 0], smpl_joints[:, 1], c='r', s=100, label='SMPL Skeleton')
	    for edge in smpl_edges:
	        joint1 = smpl_joints[edge[0]]
	        joint2 = smpl_joints[edge[1]]
	        plt.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 'k', lw=2)

	    # joint names for SMPL
	    for idx, (x, y,z) in enumerate(smpl_joints):
	        plt.text(x+0.01, y+0.01, str(idx), fontsize=12, ha='center', va='center', color='black')

	    # trans = np.array([2.0, 0.5, 0.0])
	    rabit_joints = rabit_joints*scale + trans

	    # plotting for rabit
	    plt.scatter(rabit_joints[:, 0], rabit_joints[:, 1], c='b', s=100, label='RaBit Skeleton')
	    for edge_r in rabit_edges:
	        joint1_r = rabit_joints[edge_r[0]]
	        joint2_r = rabit_joints[edge_r[1]]
	        plt.plot([joint1_r[0], joint2_r[0]], [joint1_r[1], joint2_r[1]], 'k', lw=2)

	    # joint names for Rabit
	    for idx, (x, y,z) in enumerate(rabit_joints):
	        plt.text(x+0.01, y+0.01, str(idx), fontsize=12, ha='center', va='center', color='black')


	    for (smpl_idx, rabit_idx) in corresp:
	        plt.plot([smpl_joints[smpl_idx, 0], rabit_joints[rabit_idx, 0]],
	                [smpl_joints[smpl_idx, 1], rabit_joints[rabit_idx, 1]],
	                color='green', linestyle='--', linewidth=2)
	    
	    plt.xlabel('X')
	    plt.ylabel('Y')
	    plt.legend()
	    plt.grid(True)
	    plt.legend()
	    plt.show()


