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



# loading SMPL and returning joint locations
def load_smpl(filename):

    with open(filename, 'rb') as f:
        smpl_model = pickle.load(f, encoding='latin1')

    body = SMPL(filename)
    vs = tf.expand_dims(body.template_vertices,0)
    joint_locations_local = tf.stack(values=[
                tf.matmul(vs[:, :, 0], body.joint_regressor),
                tf.matmul(vs[:, :, 1], body.joint_regressor),
                tf.matmul(vs[:, :, 2], body.joint_regressor)],axis=2, name="joint_locations_local")
    betas = np.zeros(10,dtype=np.float32)
    shape = tf.reshape(betas,[-1,10])

    joints = np.array(joint_locations_local[0])

    return joints

# loading RaBit and return the joint locations
def load_rabit(filename):

    rabit_joints = np.load(filename)

    return rabit_joints