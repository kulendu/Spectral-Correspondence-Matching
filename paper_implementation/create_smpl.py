import numpy as np 
import polyscope as ps
from utils.joints import load_smpl



'''

(0, 1): length: 0.11504147946834564
(0, 2): length: 0.11310230195522308
(0, 3): length: 0.11221449822187424
(2, 5): length: 0.38458213210105896
(5, 8): length: 0.40096551179885864
(8, 11): length: 0.13481944799423218
(1, 4): length: 0.37678787112236023
(4, 7): length: 0.40058276057243347
(7, 10): length: 0.13430216908454895
(3, 6): length: 0.13529615104198456
(6, 9): length: 0.05873068422079086
(9, 14): length: 0.14932163059711456
(9, 13): length: 0.14900162816047668
(13, 16): length: 0.0963524729013443
(16, 18): length: 0.26137229800224304
(18, 20): length: 0.24939829111099243
(20, 22): length: 0.08575001358985901
(14, 17): length: 0.10179170966148376
(17, 19): length: 0.25499144196510315
(19, 21): length: 0.2554769217967987
(21, 23): length: 0.08546747267246246
(12, 15): length: 0.0829717218875885
(9, 12): length: 0.21813981235027313
(12, 15): length: 0.0829717218875885


'''
SMPL_PATH = '../SMPL_NEUTRAL.pkl'

EDGES_SMPL = np.array([(0,1), (0,2), (0,3), (2,5), (5,8), (8,11), (1,4), (4,7),(7,10),
              (3,6), (6,9), (9,14), (9,13), (13,16), (16,18), (18, 20), (20,22),
              (14,17), (17,19), (19,21), (21, 23), (12, 15), (9, 12), (12, 15)])

def create_smpl(filename):
	joints = load_smpl(filename)
	bone_length = []

	desired_bone_length = {
		(21, 23): 0.5,
		(20, 22): 0.5,
		(0,1): 0.3,
		(0,2): 0.3,
		(8,11): 0.2,
		(7,10): 0.2, 
		(12, 15): 1.2
	}

	print(f"Original joints: {joints}")

	joints_copy = joints.copy()

	for parent, child in EDGES_SMPL:
		length = np.linalg.norm(joints[parent] - joints[child])
		bone_length.append(length)



	for (start_idx, end_idx), desired_length in desired_bone_length.items():

	    start_joint = joints_copy[start_idx]  # Start joint position
	    end_joint = joints_copy[end_idx]      # End joint position
	    
	    # Calculate the current length of the bone
	    current_length = np.linalg.norm(end_joint - start_joint)
	    
	    # Calculate the stretch factor
	    stretch_factor = desired_length / current_length
	    
	    # Update the end joint position
	    joints_copy[end_idx] = start_joint + stretch_factor * (end_joint - start_joint)
		
		# breakpoint()



	# ps.init()
	# ps.register_point_cloud("joints_copy", (joints_copy + [2.0, 0.0, 0.0]))
	# ps.register_point_cloud("joints", joints)
	# ps.show()

	return joints_copy


