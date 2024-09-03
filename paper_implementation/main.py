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
from scipy.linalg import eigh
import polyscope as ps
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.sparse.csgraph as csgraph

from utils.joints import load_smpl, load_rabit
from utils.visualiszer import Visualizer
from create_smpl import create_smpl


np.set_printoptions(suppress=True) # remove scientific notations for floating numbers

SMPL_PATH = '../SMPL_NEUTRAL.pkl'
RABIT_PATH = '../rabit_joints.npy'

SMPL_MESH = '../rest.obj'
RABIT_MESH = '../rabit.obj'

# one-to-one mapping for SMPL to rabit
DENSE_CORRESP = np.array([(0,1), (1,11), (2,13), (3,2), (4,12), (5,14), (7,15), (8,22), (9,3), (10,16), (11,23), (12, 9), (13, 4),
                 (14,17), (15,10), (16,5), (17,18), (18,6), (19,19), (20,7), (21,20), (22,8), (23,21)])

EDGES_SMPL = np.array([(0,1), (0,2), (0,3), (2,5), (5,8), (8,11), (1,4), (4,7),(7,10),
              (3,6), (6,9), (9,14), (9,13), (13,16), (16,18), (18, 20), (20,22),
              (14,17), (17,19), (19,21), (21, 23), (12, 15), (9, 12), (12, 15)])

EDGES_RABIT = np.array([
        (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), 
        (9, 3), (10, 9), (11, 1), (12, 11), (13, 1), (14, 13), (15, 12),
        (16, 15), (17, 3), (18, 17), (19, 18), (20, 19), (21, 20), (22, 14), (23, 22)
    ])


ps.init()
ps.set_ground_plane_mode("shadow_only")


joints = load_smpl(SMPL_PATH)
rabit_joints = load_rabit(RABIT_PATH)
joints_copy = create_smpl(SMPL_PATH)



# calculate the Laplacian for meshes
def mesh_laplacian():
    smpl = trimesh.load_mesh(SMPL_MESH)
    rabit = trimesh.load_mesh(RABIT_MESH)

    smpl_verts = smpl.vertices
    smpl_edges = smpl.edges
    rabit_verts = rabit.vertices
    rabit_edges = rabit.edges

    print(f"SMPL Vertices: {len(smpl_verts)}\n")
    print(f"RaBit Vertices: {len(rabit_verts)}\n")
    # ps.register_curve_network("SMPL", smpl_verts+[0, -0.4, 0], smpl_edges)
    # ps.register_curve_network("RaBit", (rabit_verts+[1, -0.3, 0])*2.0, rabit_edges)
    # ps.show()

    # for SMPL
    nodes = len(smpl_verts)
    adj_matrix = np.zeros((nodes, nodes)) # adjacency matrix

    for (i, j) in tqdm(smpl_edges):
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  

    adj_sparse = sp.csr_matrix(adj_matrix) # adjacency sparse matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1)) # deg matrix
    degree_sparse = sp.csr_matrix(degree_matrix) # deg sparse matrix

    laplacian_sparse = degree_sparse - adj_sparse # L = D - A
    laplacian_dense = laplacian_sparse.toarray()

    # eigen value and eigen vectors of a Laplacian
    print(f"Adj Matrix fro SMPL Mesh: {adj_matrix} and \n Sparse Matrix for SMPL mesh: {adj_sparse}")
    print("----Calculating Eigenvalues for SMPL -----")
    eigenvalues_smpl_mesh, eigenvectors_smpl_mesh = eigh(laplacian_dense)
    print(f"EigenValues on the verts: {len(eigenvalues_smpl_mesh)}, and Eigen vectors: {len(eigenvectors_smpl_mesh)}")



    # for RaBit
    nodes_r = len(rabit_verts)
    adj_matrix_r = np.zeros((nodes_r, nodes_r)) # adjacency matrix

    for (x, y) in tqdm(rabit_edges):
        adj_matrix_r[x, y] = 1
        adj_matrix_r[y, x] = 1  

    adj_sparse_r = sp.csr_matrix(adj_matrix_r) # adjacency sparse matrix
    degree_matrix_r = np.diag(np.sum(adj_matrix_r, axis=1)) # deg matrix
    degree_sparse_r = sp.csr_matrix(degree_matrix_r) # deg sparse matrix

    laplacian_sparse_r = degree_sparse_r - adj_sparse_r
    laplacian_dense_r = laplacian_sparse_r.toarray()
    print("----Calculating Eigenvalues for rabit-----")
    eigenvalues_rabit_mesh, eigenvectors_rabit_mesh = eigh(laplacian_dense_r)
    # print(f"EigenValues of Rabit mesh: {eigenvalues_rabit_mesh}, and Eigen vectors of Rabit mesh: {eigenvectors_rabit_mesh}")

    # skel_visualizer()
    # render_eign(eigenvectors_smpl_mesh, eigenvalues_smpl_mesh)

    return eigenvalues_smpl_mesh, eigenvectors_smpl_mesh, eigenvalues_rabit_mesh, eigenvectors_rabit_mesh, smpl_verts, rabit_verts




def laplacian():

    # for SMPL
    nodes = len(joints)
    adj_matrix = np.zeros((nodes, nodes)) # adjacency matrix
    for (i, j) in EDGES_SMPL:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  

    adj_sparse = sp.csr_matrix(adj_matrix) # adjacency sparse matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1)) # deg matrix
    degree_sparse = sp.csr_matrix(degree_matrix) # deg sparse matrix

    laplacian_sparse = degree_sparse - adj_sparse # L = D - A
    laplacian_dense = laplacian_sparse.toarray()

    # eigen value and eigen vectors of a Laplacian
    print(f"Adj Matrix: {adj_matrix} and \n Sparse Matrix: {adj_sparse}")
    eigenvalues_smpl, eigenvectors_smpl = eigh(laplacian_dense)
    print(f"EigenValues : {len(eigenvalues_smpl)}, and Eigen vectors: {len(eigenvectors_smpl)}")



    # for RaBit
    nodes_r = len(rabit_joints)
    adj_matrix_r = np.zeros((nodes_r, nodes_r)) # adjacency matrix
    for (x, y) in EDGES_RABIT:
        adj_matrix_r[x, y] = 1
        adj_matrix_r[y, x] = 1  

    adj_sparse_r = sp.csr_matrix(adj_matrix_r) # adjacency sparse matrix
    degree_matrix_r = np.diag(np.sum(adj_matrix_r, axis=1)) # deg matrix
    degree_sparse_r = sp.csr_matrix(degree_matrix_r) # deg sparse matrix

    laplacian_sparse_r = degree_sparse_r - adj_sparse_r
    laplacian_dense_r = laplacian_sparse_r.toarray()

    eigenvalues_rabit, eigenvectors_rabit = eigh(laplacian_dense_r)
    print(f"EigenValues : {eigenvalues_rabit}, and Eigen vectors: {eigenvalues_rabit}")


    return eigenvalues_smpl, eigenvectors_smpl, eigenvalues_rabit, eigenvectors_rabit, laplacian_dense, laplacian_dense_r, adj_sparse
 

# computing the HKS for SMPL and Rabit joints
def compute_hks(evals, evecs, count): # count: Scales of Diffusion 

    print(f"evals: {type(evals)} \n evecs = {type(evecs)}")
    
    evals = torch.from_numpy(evals).type(torch.float32)
    evecs = torch.from_numpy(evecs).type(torch.float32)
    print(f"evals: {type(evals)} \n evecs = {type(evecs)}")
    scales = torch.logspace(-2, 0., steps=count, dtype=torch.float32) # scales (timestep) = 16
    print(f"Shape of scales = {scales.shape}")

    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    print(f"evals: {type(evals)} \n evecs = {type(evecs)} \n scales: {type(scales)}")
    print(f"Scales: {scales}")
    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # shape: [1, 1, 16, 24]
    terms = power_coefs * (evecs * evecs).unsqueeze(2) # shape: [1, 24, 16, 24]
    
    out = torch.sum(terms, dim=-1) 

    print(f"Dimensions of the HKS: {out.shape}")

    return out


# computing HKS on the mesh
def compute_mesh_hks(evals, evecs, count): # count: Scales of Diffusion 

    print(f"evals: {type(evals)} \n evecs = {type(evecs)}")
    
    evals = torch.from_numpy(evals).type(torch.float32)
    evecs = torch.from_numpy(evecs).type(torch.float32)
    print(f"evals: {type(evals)} \n evecs = {type(evecs)}")
    scales = torch.logspace(-2, 0., steps=count, dtype=torch.float32) # scales (timestep) = 16
    print(f"Shape of scales = {scales.shape}")

    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    print(f"evals: {type(evals)} \n evecs = {type(evecs)} \n scales: {type(scales)}")
    print(f"Scales: {scales}")
    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # shape: [1, 1, 16, 24]
    terms = power_coefs * (evecs * evecs).unsqueeze(2) # shape: [1, 24, 16, 24]
    
    out_hks_mesh = torch.sum(terms, dim=-1) 

    print(f"Dimensions of the HKS: {out.shape}")

    return out_hks_mesh


# M for meshes
def adjacency_matrix(k, f_dim, eval_smpl, evec_smpl, eval_rabit, evec_rabit, joints, joints_rabit, adj_sparse):

    hks_smpl = compute_hks(eval_smpl[:k], evec_smpl[:,:k], f_dim) #(B,V,feature_dim)
    hks_rabit = compute_hks(eval_rabit[:k], evec_rabit[:,:k], f_dim) #(B,V,feature_dim)

    smpl_mat = np.zeros((24,24))
    rabit_mat = np.zeros((24,24))

    num_joints = len(joints)

    adj_mat = torch.zeros((num_joints**2, num_joints**2))
    sigma = 0.01


    print("------ Calculating the Adjacency Matrix -------")

    for i in range(num_joints):
        for j in range(num_joints):
            for k in range(num_joints):
                for l in range(num_joints):

                    node1 = ((hks_smpl[0][i] - hks_rabit[0][j])**2).mean()
                    node2 = ((hks_smpl[0][k] - hks_rabit[0][l])**2).mean()

                    if i==k and j==l: # checking the diagonal entries: M(a,a)

                        mat_val = np.exp(-((node1)**2)/(2*sigma**2))
                        adj_mat[(i*num_joints)+j, (k*num_joints)+l] = mat_val
                           
                    else:
                        adj_mat[(i*num_joints)+j, (k*num_joints)+l] = np.exp(-((node1-node2)**2)/(2*sigma**2))


    adj_mat = adj_mat.numpy()
    print(f"Shape of Adjacency Matrix: {adj_mat.shape}")

    print("-----")

    return adj_mat



# M for skeletons
def adjacency_matrix(k, f_dim, eval_smpl, evec_smpl, eval_rabit, evec_rabit, joints, joints_rabit, adj_sparse):

    hks_smpl = compute_hks(eval_smpl[:k], evec_smpl[:,:k], f_dim) #(B,V,feature_dim)
    hks_rabit = compute_hks(eval_rabit[:k], evec_rabit[:,:k], f_dim) #(B,V,feature_dim)

    smpl_mat = np.zeros((24,24))
    rabit_mat = np.zeros((24,24))

    num_joints = len(joints)

    adj_mat = torch.zeros((num_joints**2, num_joints**2))
    sigma = 0.01


    print("------ Calculating the Adjacency Matrix -------")

    for i in range(num_joints):
        for j in range(num_joints):
            for k in range(num_joints):
                for l in range(num_joints):

                    node1 = ((hks_smpl[0][i] - hks_rabit[0][j])**2).mean()
                    node2 = ((hks_smpl[0][k] - hks_rabit[0][l])**2).mean()

                    if i==k and j==l: # checking the diagonal entries: M(a,a)

                        mat_val = np.exp(-((node1)**2)/(2*sigma**2))
                        adj_mat[(i*num_joints)+j, (k*num_joints)+l] = mat_val
                           
                    else:
                        adj_mat[(i*num_joints)+j, (k*num_joints)+l] = np.exp(-((node1-node2)**2)/(2*sigma**2))


    adj_mat = adj_mat.numpy()
    print(f"Shape of Adjacency Matrix: {adj_mat.shape}")

    print("-----")

    return adj_mat



def compute_principal_eigenvector(M):

    eigenvalues, eigenvectors = eigh(M)
    principal_eigenvector_index = np.argmax(eigenvalues)
    return eigenvectors[:, principal_eigenvector_index]



def perform_assignment(M, x_star, num_joints):

    '''
    x_star: principle eigenvector of M
    L: initialized with all possible one-to-one correspondences
    x: solution vector (starts with 0, when the assignment is selected from L, that particular idx == 1)
    a_star: Best assignment in the current iteration (if the best value is (0,0) from L in the first iteration based on the 
    highest value in x_star, then set a_star = (0,0))
    a_star_index: index in x_star that corresponds to the assignment a_star
    '''

    x = np.zeros((num_joints * num_joints), dtype=int)
    L = [(i, j) for i in range(num_joints) for j in range(num_joints)]

    while L:
        a_star = None
        max_value = -float('inf')

        for index, a in enumerate(L):
            a_index = (a[0] * num_joints) + a[1]
            value = x_star[a_index]

            if value > max_value:
                max_value = value
                a_star = a

        np.set_printoptions(suppress=True)
        print(f"a_star: {a_star} and max_value: {max_value}")
        
        a_star_index = (a_star[0] * num_joints) + a_star[1]

        if x_star[a_star_index] == 0: # if the index of x_star that corresponds to the assignment a_star == 0
            break
        
        x[a_star_index] = 1
        L.remove(a_star)
        

        # Remove conflicting assignments
        i, i_bar = a_star
        updated_L = []
        
        for assignment in L:
            j, j_bar = assignment
            if i != j and i_bar != j_bar:
                updated_L.append(assignment)
        
        L = updated_L


    return x


def new_perform_assignment(M, x_star, num_joints):
    '''
    x_star: principle eigenvector of M
    L: initialized with all possible one-to-one correspondences
    x: solution vector (starts with 0, when the assignment is selected from L, that particular idx == 1)
    a_star: Best assignment in the current iteration (if the best value is (0,0) from L in the first iteration based on the 
    highest value in x_star, then set a_star = (0,0))
    a_star_index: index in x_star that corresponds to the assignment a_star
    '''
    x = np.zeros((num_joints * num_joints), dtype=int)
    L = [(i, j) for i in range(num_joints) for j in range(num_joints)]
    candidate_assignment = [(i, j) for i in range(num_joints) for j in range(num_joints)]

    while L:

        max_index = np.argmax(x_star) 
        a_star = candidate_assignment[max_index]

        print(f"Selected a_star: {a_star} with x_star value: {x_star[max_index]}")

        
        if x_star[max_index] == (-float('inf')):
            print("x_star value is 0. Exiting.")
            break

        # Update the solution vector x
        x[max_index] = 1
        x_star[max_index] = (-float('inf'))

        # Remove conflicting assignments
        i, i_bar = a_star

        # marking conflicts in x_star
        for index, (j, j_bar) in enumerate(candidate_assignment):
            if (i == j or i_bar == j_bar) and x_star[index] != -float('inf'):
                x_star[index] = -float('inf')

        # breakpoint()
        
        new_L = []
        # removing the conflicts
        for assignment in L:
            j, j_bar = assignment
            if i != j and i_bar != j_bar:
                new_L.append(assignment)
        L = new_L        


        final_corresp = []

        for index, value in enumerate(x):
            if value == 1:
                final_corresp.append(candidate_assignment[index])

    # breakpoint()


    return x, final_corresp



def extract_corresp(x, num_joints):
    correspondences = []

    for a_star_index, value in enumerate(x):

        if value == 1:

            i = a_star_index // num_joints
            i_bar = a_star_index % num_joints
            correspondences.append((i, i_bar))

    print(f"correspondences: {correspondences}")

    return correspondences



# def plot_correspondences(correspondences, smpl_joints, rabit_joints):
#     plt.figure()

#     smpl_x, smpl_y, smpl_z = zip(*smpl_joints)
#     rabbit_x, rabbit_y, rabbit_z = zip(*rabit_joints)
    
#     plt.scatter(smpl_x, smpl_y, c='blue', label='SMPL')
#     plt.scatter(rabbit_x, rabbit_y, c='red', label='Rabbit')

#     for i, i0 in correspondences:
#         plt.plot([smpl_joints[i][0], rabit_joints[i0][0]], [smpl_joints[i][1], rabit_joints[i0][1]], 'k--')
    
#     plt.legend()
#     plt.show()







if __name__ == '__main__':

    K=24 #number of eigenvectors to consider
    feature_dim = 16 # timesteps (t) for the heat diffusion
    num_joints = len(joints)

    eigenvalues_smpl_mesh, eigenvectors_smpl_mesh, eigenvalues_rabit_mesh, eigenvectors_rabit_mesh, smpl_verts, rabit_verts = mesh_laplacian()

    hks_vis = Visualizer.plot_hks_corresp(K, feature_dim, eigenvalues_smpl_mesh, eigenvectors_smpl_mesh, eigenvalues_rabit_mesh, eigenvectors_rabit_mesh, smpl_verts, rabit_verts)
    breakpoint()
    eigenvalues_smpl, eigenvectors_smpl, eigenvalues_rabit, eigenvectors_rabit, laplacian_dense, laplacian_dense_r, adj_sparse = laplacian()
 

    M = adjacency_matrix(K, feature_dim, eigenvalues_smpl, eigenvectors_smpl, eigenvalues_rabit, eigenvectors_rabit, joints, rabit_joints, adj_sparse)
    # M = adjacency_matrix(K, feature_dim, eigenvalues_smpl, eigenvectors_smpl, eigenvalues_smpl, eigenvectors_smpl, joints, joints, adj_sparse)

    x_star = compute_principal_eigenvector(M)
        
    x, final_corresp = new_perform_assignment(M, x_star, num_joints)
    # breakpoint()

    # correspondences = extract_corresp(x, num_joints)
    plot = Visualizer.plot_corresp(final_corresp, joints, rabit_joints, EDGES_SMPL, EDGES_RABIT, scale=3.5, trans=[2,0,0])

    # plot_correspondences(correspondences, joints, rabit_joints)


