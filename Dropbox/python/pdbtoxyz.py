#converts a PDB file to a semi-colon separated fixed length xyz file


import os
import argparse
import numpy as np




#command line parameters
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', dest='pdb_dir', action='store', help='directory with PDB files')
parser.add_argument('--file', dest='pdb_filename', action='store', help='a single PDB file')
parser.add_argument('--max_files', dest='max_files', action='store', help='max numbers of PDB files to process')
args = parser.parse_args()



#extracts only the coordinates from a PDB file
#write the coordinates separated by ; to a file which ends with .xyz
def pdb_to_xyz(pdb_filename):

	separator = ';'
	pdb_file = open(pdb_filename, 'rU')
	xyz_file = open(pdb_filename + '.xyz', 'w')
	
	for line in pdb_file.readlines():
		if line[0:4] == 'ATOM':
			if line[13:14] != 'H':
				#if line[13:15] == 'CA':
				xyz_file.write(line[33:38] + separator + line[41:46] + separator + line[49:54] + '\n')

	pdb_file.close()
	xyz_file.close()
	return True

def read_xyz(xyz_filename):
	
	coordinates = []
	xyz_file = open(xyz_filename, 'rU')
	for line in xyz_file.readlines():
		coordinates.append([float(line[0:5]), float(line[6:11]), float(line[12:17])])
	xyz_file.close()
	return np.array(coordinates)
	
	
def fit(P, Q):
	#"""
	#Varies the distance between P and Q, and optimizes rotation for each step
	#until a minimum is found.
	#"""
	step_size = P.max(0)
	threshold = step_size*1e-9
	rmsd_best = kabsch_rmsd(P, Q)
	while True:
		for i in range(3):
			temp = np.zeros(3)
			temp[i] = step_size[i]
			rmsd_new = kabsch_rmsd(P+temp, Q)
			if rmsd_new < rmsd_best:
				rmsd_best = rmsd_new
				P[:, i] += step_size[i]
			else:
				rmsd_new = kabsch_rmsd(P-temp, Q)
				if rmsd_new < rmsd_best:
					rmsd_best = rmsd_new
					P[:, i] -= step_size[i]
				else:
					step_size[i] /= 2
		if (step_size <= threshold).all():
			break
	return rmsd_best

#######################
##https://github.com/charnley/rmsd/blob/master/calculate_rmsd
def kabsch_rmsd(P, Q):
	#"""
	#Rotate matrix P unto Q and calculate the RMSD
	#"""
	P = rotate(P, Q)
	return rmsd(P, Q)


def rotate(P, Q):
	#"""
	#Rotate matrix P unto matrix Q using Kabsch algorithm
	#"""
	U = kabsch(P, Q)
	
	# Rotate P
	P = np.dot(P, U)
	return P
def kabsch(P, Q):
	#"""
	#The optimal rotation matrix U is calculated and then used to rotate matrix
	#P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
	#calculated.
	#Using the Kabsch algorithm with two sets of paired point P and Q,
	#centered around the center-of-mass.
	#Each vector set is represented as an NxD matrix, where D is the
	#the dimension of the space.
	#The algorithm works in three steps:
	#- a translation of P and Q
	#- the computation of a covariance matrix C
	#- computation of the optimal rotation matrix U
	#http://en.wikipedia.org/wiki/Kabsch_algorithm
	#Parameters:
	#P -- (N, number of points)x(D, dimension) matrix
	#Q -- (N, number of points)x(D, dimension) matrix
	#Returns:
	#U -- Rotation matrix
	#"""
	
	# Computation of the covariance matrix
	C = np.dot(np.transpose(P), Q)
	
	# Computation of the optimal rotation matrix
	# This can be done using singular value decomposition (SVD)
	# Getting the sign of the det(V)*(W) to decide
	# whether we need to correct our rotation matrix to ensure a
	# right-handed coordinate system.
	# And finally calculating the optimal rotation matrix U
	# see http://en.wikipedia.org/wiki/Kabsch_algorithm
	V, S, W = np.linalg.svd(C)
	d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
	
	if d:
		S[-1] = -S[-1]
		V[:, -1] = -V[:, -1]
	
	# Create Rotation matrix U
	U = np.dot(V, W)
	
	return U
def rmsd(V, W):
	#"""
	#Calculate Root-mean-square deviation from two sets of vectors V and W.
	#"""
	D = len(V[0])
	N = len(V)
	rmsd = 0.0
	for v, w in zip(V, W):
		rmsd += sum([(v[i]-w[i])**2.0 for i in range(D)])
	return np.sqrt(rmsd/N)

def centroid(X):
	"""
	Calculate the centroid from a vectorset X
	"""
	C = sum(X)/len(X)
	return C
#######################

#main program
if __name__ == "__main__":
	
	pdb_filename = args.pdb_filename
	pdb_dir = args.pdb_dir
	max_files = int(args.max_files)
	print (max_files)
	if max_files == None or max_files == 0:
		args.max_files = -1
		max_files = -1

	pdb_files = [] #stores all the pdb files which have to be parsed
	xyz_files = []
	
	if pdb_filename != None:
		pdb_files.append(pdb_filename)
	elif pdb_dir != None:
		for filename in os.listdir(pdb_dir):
			if filename.endswith('.pdb'):
				pdb_files.append(pdb_dir + "/" + filename)
				max_files += -1
				if max_files == 0:
					break
	else:
		pdb_files=[]
	
	max_files = int(args.max_files)
	for pdb_file in pdb_files:
		print (pdb_file)
		if pdb_file.endswith('.pdb'):
			if os.path.isfile(pdb_file + '.xyz') == False:
				pdb_to_xyz(pdb_file)
				max_files += -1
				if max_files == 0:
					break

	for pdb_file in pdb_files:
		if os.path.isfile(pdb_file + '.xyz') == True:
			xyz_files.append(pdb_file + '.xyz')

	ref_file = '/home/ashafix/data/downloads/virtualmachine/1ubq_nohomologs/decoys/S_00000001_1.pdb'
	pdb_to_xyz(ref_file)
	ref_coordinates = read_xyz(ref_file + '.xyz')
	ref_coordinates -= centroid(ref_coordinates)
	
	max_files = int(args.max_files)
	for xyz_file in xyz_files:
		coordinates = (read_xyz(xyz_file))
		
		#centers coordinate array
		#centroid = np.mean(coordinates[:,-3:], axis = 0)
		
		#coordinates = np.subtract(coordinates, centroid)
		coordinates -= centroid(coordinates)
	
		print ("Normal RMSD:", rmsd(coordinates, ref_coordinates))
		print ("Kabsch RMSD:", kabsch_rmsd(coordinates, ref_coordinates))
		print ("Fitted RMSD:", fit(coordinates, ref_coordinates))
		max_files += -1
		if max_files == 0:
			break


	print ('\nfinished')
	
