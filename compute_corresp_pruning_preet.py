import numpy as np
import pickle
import polyscope as ps
import matplotlib.pyplot as plt
import scipy.linalg
from smpl import SMPL
import torch
import tensorflow as tf
import scipy
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from scipy.linalg import eigh
import vedo
import trimesh
import networkx as nx 
from tqdm import tqdm


SMPL_PATH = './SMPL_NEUTRAL.pkl'
RABIT_PATH = './joints_list.pkl'

EDGES_SMPL = [(0,1), (0,2), (0,3), (2,5), (5,8), (8,11), (1,4), (4,7),(7,10),
              (3,6), (6,9), (9,14), (9,13), (13,16), (16,18), (18, 20), (20,22),
              (14,17), (17,19), (19,21), (21, 23), (12, 15), (9, 12)]

EDGES_RABIT = [
        (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), 
        (9, 3), (10, 9), (11, 1), (12, 11), (13, 1), (14, 13), (15, 12),
        (16, 15), (17, 3), (18, 17), (19, 18), (20, 19), (21, 20), (22, 14), (23, 22)
    ]


ps.init()
# ps.set_background_color([0.0, 0.0, 0.0])
ps.set_ground_plane_mode("shadow_only")



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


    for i in enumerate(joint_locations_local[0]):
        print(i)

    print(f"Joints : {joint_locations_local}")
    joints = np.array(joint_locations_local[0])
    

    # plt.scatter(joints[:,0], joints[:,1])
    
    # for i in range(joints.shape[0]):
    #     plt.text(joints[i,0], joints[i,1], str(i))

    # plt.show()

    return joints


def load_rabit(filename):
    rabit_joints = np.load(filename,allow_pickle=True)
    print(f"Length of RaBit joints: {len(rabit_joints)}")

    return rabit_joints


def skel_visualizer():
    joints = load_smpl(SMPL_PATH)
    rabit_joints = load_rabit(RABIT_PATH)

    edges_smpl = np.array(EDGES_SMPL)
    edges_rabit = np.array(EDGES_RABIT)
    
    smpl_joints = ps.register_point_cloud("SMPL joints", joints)
    skel = ps.register_curve_network("Connections", joints, edges_smpl)
    r_joints = ps.register_point_cloud("Rabit Joints", rabit_joints)
    skel_r = ps.register_curve_network("rabt skel", ((rabit_joints + [0.6, 0.0, 0.0])* 4.0), edges_rabit)

    ps.show()

# renderer using vedo
def render_eign(eigenvectors, eigenvalues):
    if eigenvectors.shape[1] < 5:
        raise ValueError("Need at least three eigenvectors for 3D visualization")

    points = eigenvectors[:, 1:4]

    point_cloud = vedo.Points(points, r=15, alpha=1)

    point_cloud.cmap('viridis', eigenvalues, on='points')

    plotter = vedo.Plotter()
    plotter.show(point_cloud, "3D Visualization of Eigenvalues and Eigenvectors")


def laplacian():
    joints = load_smpl(SMPL_PATH)
    rabit_joints = load_rabit(RABIT_PATH)

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

    # skel_visualizer()
    # render_eign(eigenvectors_smpl, eigenvalues_smpl)

    return eigenvalues_smpl, eigenvectors_smpl, eigenvalues_rabit, eigenvectors_rabit, laplacian_dense, laplacian_dense_r, adj_sparse

    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(eigenvalues_smpl, 'bo-', markersize=5)
    # plt.title('Eigenvalues of Laplacian')
    # plt.xlabel('Index')
    # plt.ylabel('Eigenvalue')
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # num_eigenvectors_to_plot = 4
    # for i in range(num_eigenvectors_to_plot):
    #     plt.plot(eigenvectors_smpl[:, i], label=f'Eigenvector {i}')
    # plt.title('Eigenvectors of Laplacian')
    # plt.xlabel('Node')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # cax1 = ax1.imshow(laplacian_dense, cmap='viridis', interpolation='none')
    # ax1.set_title('Laplacian for SMPL')
    # # ax1.set_xlabel('Node')
    # # ax1.set_ylabel('Node')
    # fig.colorbar(cax1, ax=ax1, label='Value')

    # cax2 = ax2.imshow(laplacian_dense_r, cmap='viridis', interpolation='none')
    # ax2.set_title('Laplacian for RaBit')
    # # ax2.set_xlabel('Node')
    # # ax2.set_ylabel('Node')
    # fig.colorbar(cax2, ax=ax2, label='Value')

    # plt.tight_layout()
    # plt.show()


def compute_hks(evals, evecs, count):
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

def visualize_corresp(k, f_dim, eval_smpl, evec_smpl, eval_rabit, evec_rabit, j_smpl, j_rabit):

    hks_smpl = compute_hks(eval_smpl[:k], evec_smpl[:,:k], f_dim) # (B,V,feature_dim)
    hks_rabit = compute_hks(eval_rabit[:k], evec_rabit[:,:k], f_dim) # (B,V,feature_dim)
    
    color_smpl = hks_smpl[0].numpy()[:,:3]
    color_smpl = (color_smpl - color_smpl.min())/(color_smpl.max()-color_smpl.min())
    color_rabit = hks_rabit[0].numpy()[:,:3]
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



'''
TO-DOs:
1. Compute the Degree of each node
2. Match the deg(smpl) and deg(rabit) | deg(n1) = deg(n2)
3. Compute the A: Association Graph, where nodes will be (n1,n2), where deg(n1) == deg(n2) [Pruned Graph]
4. Edge weight will be calculated as the HKS/geodesic distance (geo. distance = edges covered) 
5. Now, find the clique of this Graph
'''
def degree(edges,v):
    for i,j in edges:
        v[i]+=1
        v[j]+=1

def degree_match(a,b):
    r=0
    c=0
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i]==b[j]:
                # d[r][c]=i
                # d[r][c+1]=j
                r+=1
                
    e=np.zeros((r,2))
    h=0
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i]==b[j]:
                e[h][c]=i
                e[h][c+1]=j
                h+=1
                c=0
    return e

# def asso(d,joints1,joints2):
#     ass=np.zeros(304,304)
#     for i in range(304):
#         for j in range(304):
#             i1=d[i][0]
#             i2=d[i][1]
#             i3=d[j][0]
#             i4=d[j][1]
#             d1=(joints1[i][0]-joints1[j][0])**2+(joints1[i][1]-joints1[j][1])**2+(joints1[i][2]-joints1[j][2])**2
#             d2=(joints2[i][0]-joints2[j][0])**2+(joints2[i][1]-joints2[j][1])**2+(joints2[i][2]-joints2[j][2])**2
#             score=np.exp()
# feature-wise geodesic distance
def compute_association_mat(k, f_dim, eval_smpl, evec_smpl, eval_rabit, evec_rabit, joints, joints_rabit, adj_sparse,d):
    hks_smpl = compute_hks(eval_smpl[:k], evec_smpl[:,:k], f_dim) #(B,V,feature_dim)
    hks_rabit = compute_hks(eval_rabit[:k], evec_rabit[:,:k], f_dim) #(B,V,feature_dim)

    smpl_mat = np.zeros((24,24))
    rabit_mat = np.zeros((24,24))

    print(smpl_mat.shape)
    print(rabit_mat.shape)

    # initialize the Graph
    G_SMPL = nx.Graph()
    for idx, j in enumerate(joints):
        G_SMPL.add_node(idx)
    G_SMPL.add_edges_from(EDGES_SMPL) 


    G_RABIT = nx.Graph()
    for idx_r, j_r in enumerate(joints_rabit):
        G_RABIT.add_node(idx_r)
    G_RABIT.add_edges_from(EDGES_RABIT)    

    # graphs of the joints (nodes) and bones (edges)
    # subax1 = plt.subplot(121)
    # nx.draw(G_SMPL, with_labels=True, font_weight='bold')
    # subax2 = plt.subplot(122)
    # nx.draw(G_RABIT, with_labels=True, font_weight='bold')
    # plt.show()

    '''Finding the degree of each nodes of SMPL and RaBit'''
    for smpl_joint in G_SMPL.nodes():
        
        print(f"Nodes of the SMPL: {smpl_joint}")


    
    for i in range(24):
        for j in range(24):
            hks_s = hks_smpl[0][i] - hks_smpl[0][j]
            hks_r = hks_rabit[0][i] - hks_rabit[0][j]
            '''Not needed at this point!'''
            smpl_mat[i][j] = np.linalg.norm(np.exp(-(hks_s**2)*(len(nx.shortest_path(G_SMPL, source=i, target=j)))))
            rabit_mat[i][j] = np.linalg.norm(np.exp(-(hks_r**2)*(len(nx.shortest_path(G_RABIT, source=i, target=j)))))

    # normalised feature matrices
    smpl_mat_norm = (smpl_mat - smpl_mat.min()) / (smpl_mat.max() - smpl_mat.min())
    rabit_mat_norm = (rabit_mat - rabit_mat.min()) / (rabit_mat.max() - rabit_mat.min())

    print(f"SMPL Feature wise geodesic distance: {smpl_mat_norm} \n For RaBit: {rabit_mat_norm}")
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # ax1.imshow(smpl_mat_norm, cmap="hot")
    # ax1.set_title("SMPL")
    # ax2.imshow(rabit_mat_norm, cmap="hot")
    # ax2.set_title("RaBit")
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    num_joints = 24
    
    # pair_nodes = torch.zeros((num_joints**2,f_dim))
    # for i in range(num_joints):
    #     for j in range(num_joints):
    #         pair_nodes[i*num_joints + j] = (hks_smpl[0][i] - hks_rabit[0][j])**2
    # pair_nodes = pair_nodes.numpy()

    # sigma_values = np.arange(0.0, 0.3, 0.001)

    # corresp_mat = torch.zeros((num_joints**2, num_joints**2))
    # norm_corresp_mat = torch.zeros((num_joints**2, num_joints**2))
    corresp_mat=np.zeros((304,304))
    for i in range(304):
        for j in range(304):
            i1=d[i][0]
            i2=d[i][1]
            i3=d[j][0]
            i4=d[j][1]

            hks_s = hks_smpl[0][int(i1)] - hks_smpl[0][int(i3)]
            hks_r = hks_rabit[0][int(i2)] - hks_rabit[0][int(i4)]
            node1=(hks_s**2).mean()
            node2=(hks_r**2).mean()
            sigma = 0.0000001
            corresp_mat[i][j]=np.exp(-((node1-node2)**2)/sigma).mean() 
            # d1=(joints1[i][0]-joints1[j][0])**2+(joints1[i][1]-joints1[j][1])**2+(joints1[i][2]-joints1[j][2])**2
            # d2=(joints2[i][0]-joints2[j][0])**2+(joints2[i][1]-joints2[j][1])**2+(joints2[i][2]-joints2[j][2])**2
            # score=np.exp()
    # for i in range(num_joints):
    #     for j in range(num_joints):
    #         for k in range(num_joints):
    #             for l in range(num_joints):
    #                 node1 = ((hks_smpl[0][i] - hks_rabit[0][j])**2).mean()
    #                 node2 = ((hks_smpl[0][k] - hks_rabit[0][l])**2).mean()
                    
    #                 sigma = 0.1
    #                 # print(f"\n Sigma Values: {sigma_values} \n")
    #                 corresp_mat[(i*num_joints)+j, (k*num_joints)+l] = np.exp(-((node1-node2)**2)/sigma).mean() 
                    # corresp_mat[(i*num_joints)+j, (k*num_joints)+l] = np.mean((smpl_mat[i][j] - rabit_mat[k][l])**2)
    row_sum=[]
    sum=0
    for i in range(304):
        sum=0
        for j in range(304):
            sum+=corresp_mat[i][j]
        row_sum.append(sum)
    print(len(row_sum),row_sum)

    indices = [idx for idx, x in enumerate(row_sum)]
    row_values = dict(zip(indices, row_sum))
    sort = {k: v for k, v in sorted(row_values.items(), key=lambda item: item[1])}
    keys_in_reverse = list(reversed(sort.keys()))
    check1=np.zeros(24)
    run1=0
    check2=np.zeros(24)
    run2=0
    for i in keys_in_reverse:
        if(check1[int(d[i][0])]==0 and run1<24 and check2[int(d[i][1])]==0 and run2<24):
            print(d[i][0],d[i][1])
            check1[int(d[i][0])]=1
            run1+=1
            check2[int(d[i][1])]=1
            run2+=1
    breakpoint()    
    # corresp_mat = corresp_mat.numpy()
    # np.fill_diagonal(corresp_mat, 0)
    # A = nx.from_numpy_array(corresp_mat)

    # affinities = []

    # for node1, node2, attr in A.edges(data=True):
    #     weight = attr['weight']
    #     # attr['affinity'] = weight
    #     affinities.append(weight)
            

    # bins = np.arange(0.0, 1.0, 0.1)
    # plt.hist(affinities, color='skyblue', bins=bins, edgecolor='black')
    # plt.xlabel('Affinity')
    # plt.ylabel('Frequency')
    # plt.show()
                                


# for i in range(num_joints**2):
#     norm_corresp_mat[i] = (corresp_mat[i] - corresp_mat[i].min()) / (corresp_mat[i].max() - corresp_mat[i].min())
    # corresp_mat = corresp_mat.numpy()
    # norm_corresp_mat = norm_corresp_mat.numpy()
    # norm_mat_corresp = (corresp_mat - corresp_mat.min()) / (corresp_mat.max() - corresp_mat.min())

    print(f"\n Correspondence Matrix: {corresp_mat} and Shape: {corresp_mat.shape}")
    np.fill_diagonal(corresp_mat, 0)
    
    plt.imshow(corresp_mat)
    plt.title("Correspondence Matrix")
    plt.colorbar()
    # plt.axis([0, num_joints**2 -1, 0, num_joints**2-1]) 
    # plt.gca().invert_xaxis()
    plt.show()

    # GG = nx.from_numpy_array(corresp_mat)
    # nx.draw(GG, with_labels=True, node_color='lightblue', edge_color='black')
    # plt.show()
    
    print("-------------------------- \n")
    
    # Creating the Association matrix
    np.fill_diagonal(corresp_mat, 0) # diagonal entries = 0
    print(f"\n Corresp matrix after removing the diaginal: {corresp_mat}")
    breakpoint()
    A = nx.from_numpy_array(corresp_mat)

    thresholds = [0.0, 0.900, 0.998, 1.0]
    threshold_ranges = []
    affinities = []

    colors = ['blue', 'green', 'red']
    


    # for i in range(len(thresholds)):
    #     threshold_ranges.append((thresholds[i], thresholds[i + 1]))

    for node1, node2, attr in A.edges(data=True):
        weight = attr['weight']
        affinities.append(weight)

    pos = nx.spring_layout(A, seed=7)
    # nx.draw_networkx_nodes(A, pos, node_size=400)

    edges = A.edges(data=True)
    
    edge_colors = []
    

    # subgraphs = []

    # for i in range(len(threshold_ranges)):
    #     low, high = threshold_ranges[i]
    #     subgraph = nx.Graph()
    #     subgraph.add_nodes_from(A.nodes())
        
    #     for u, v, data in A.edges(data=True):
    #         weight = data['weight']
    #         if low < weight <= high:
    #             subgraph.add_edge(u, v, weight=weight)
        
    #     subgraphs.append((subgraph, colors[i]))


    for node1, node2, data in edges:
        if data['weight'] == 1.0:
            edge_colors.append('blue')
        else:
            edge_colors.append('white')

    GG = nx.from_numpy_array(corresp_mat)
    nx.draw(GG, with_labels=True, node_color='lightblue', edge_color=edge_colors)
    plt.show()

    # fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    # for index, (subgraph, color) in enumerate(subgraphs):
    #     ax = axes[index]
    #     pos = nx.spring_layout(subgraph)
    #     nx.draw(subgraph, pos, with_labels=True, node_color='lightgray', edge_color=color, node_size=500, font_size=16, font_color='black', edgecolors='black', ax=ax)
    #     low, high = threshold_ranges[index]
    #     ax.set_title(f'Edges in Range ({low} - {high})')

    # plt.tight_layout()
    # plt.show()
    corresp = []
    for i in edge_colors:
        if i == 'blue':
            corresp.append(i)

    breakpoint()
    bins = np.linspace(0.0, 1.0, num=10)
    plt.hist(affinities, color='skyblue', bins=bins, edgecolor='black')
    plt.ylim(0, len(affinities))
    plt.xlabel('Affinity')
    plt.ylabel('Frequency')
    plt.show()
    # plt.savefig(f"{sigma}.png")
    # print(f"------ Saving figures for {sigma} --------")
    # plt.clf()

        # nx.draw_networkx_edges(A, pos, edge_color=edge_colors, width=2)
        # nx.draw_networkx_labels(A, pos, font_size=10, font_color="black")
        # plt.figure(figsize=(8, 6))
        # nx.draw(A, pos, with_labels=True, edge_color=edge_colors, node_size=400, font_size=10, font_color='white', edgecolors=edge_colors)
        # plt.title("Graph Visualization with Affinity-Based Edge Coloring")
        # plt.show()






    '''Getting the Degree matrix'''
    D = np.zeros_like(corresp_mat)

    for i in range(corresp_mat.shape[0]):
        row_sum = np.sum(corresp_mat[i, :])
        D[i, i] = row_sum
    print(f"Shape of the Degree Matrix: {D.shape}, \n and D: {D} ")

    L = D - corresp_mat 
    print(f"Shape of the Lapalcian Matrix: {L.shape}, \n and L: {L} ")

    # A_nx = nx.from_numpy_array(adj_sparse)
    # L_nx = nx.laplacian_matrix(A_nx).toarray()
    eigenvalues, eigenvectors = scipy.linalg.eigh(L)
    eigenvectors = eigenvectors.sort()
    fiedler_vector = eigenvectors[1]
    proj = corresp_mat @ fiedler_vector


    # partition_1 = []
    # partition_2 = []

    # for i in range(len(A_nx.nodes)):
    #     if fiedler_vector[i] >= 0:
    #         partition_1.append(A_nx.nodes[i])
    #     else:
    #         partition_2.append(A_nx.nodes[i])




if __name__ == '__main__':
    K=24 #number of eigenvectors to consider
    feature_dim = 16 # timesteps for the heat diffusion
    eigenvalues_smpl, eigenvectors_smpl, eigenvalues_rabit, eigenvectors_rabit, laplacian_dense, laplacian_dense_r, adj_sparse = laplacian()

    joints = load_smpl(SMPL_PATH)
    rabit_joints = load_rabit(RABIT_PATH)
    a=np.zeros(24)
    b=np.zeros(24)
    degree(EDGES_SMPL,a)
    degree(EDGES_RABIT,b)
    print('a,b:',a,b)
    # d1=[]
    # d2=[]
    d1=degree_match(a,b)
    d2=degree_match(b,a)
    print('d1,d2',d1,d2)
    print(f"SMPL Joints: {joints}")

    # visualizing the HKS for SMPL and RaBit
    # vis = visualize_corresp(K, feature_dim, eigenvalues_smpl, eigenvectors_smpl, 
    #                         eigenvalues_rabit, eigenvectors_rabit, joints, rabit_joints)

    corresp_mat = compute_association_mat(K, feature_dim, eigenvalues_smpl, eigenvectors_smpl, eigenvalues_rabit, eigenvectors_rabit, joints, rabit_joints, adj_sparse,d1)

    print("-------------------------- \n")
    # print(laplacian_dense)

    # load_smpl(SMPL_PATH)
    # skel = skel_visualizer()