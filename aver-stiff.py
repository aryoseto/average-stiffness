'''	This script is to read multiple linearise stiffness matrices
	from SPLICE, extract the factors, and average the stiffness matrices

		aver-stiff.py <control_file.inp>

		control_file.inp
		[0] 1st line -- Pile prefix name (e.g Pile)
		[1] 2nd line -- Linearised stiffness matrix lis file (usually SPLICE_EQSTIFF.LIS)
		[2] 3rd line -- Number of piles (integer) 
		[3] 4th line -- Output file name (extension should be excluded)




'''

import numpy as np
import os
import sys



def checkargv() :
	''' Checking argument '''
	res = 0
	if len(sys.argv) == 1:
		print('Please input the control file name with its extension')
	elif len(sys.argv) > 2:
		print('Only accept 1 argument')
	elif len(sys.argv) == 2:
		if os.path.exists(sys.argv[1]): #Check if the path is valid
			res = 1
		else :
			print('File does not exists')

	return res


def to_list(nameoffile, stripstatus):
	''' Function to read lines into a list '''
	with open(nameoffile, 'r') as f:
        #llist = [line.strip() for line in f] 
		if stripstatus == True :
			llist = [line.strip() for line in f]
		else :
			llist = [line for line in f]
	return llist


def line_num(nameoflist, word, start, end):
	''' Function to return line number that has specific word'''
	
	linenumber = []
	for i, item in enumerate(nameoflist):
		if word in item[start : end]:
			linenumber.append(i)
	return linenumber

def pile_head_coord(input_list):
	''' Function to extract pile head coordinate '''
	
	# Get line number where the coordinate start (list)
	start_line = line_num(input_list, "SUMMARY OF PILE GEOMETRY VALUES", 0, 40)

	#print(start_line)

	coordinate = []
	extract_status = True

	# for i, line in enumerate(input_list):
	# 	if  start_line[0] + 4 < i <= start_line[0] + 4 + no_of_pile :
	# 		coordinate.append(line.split())

	for i, line in enumerate(input_list):
		if i > start_line[0] + 4 and extract_status :
			if len(line.strip()) == 0 :
				extract_status = False
			else:
				coordinate.append(line.split()) 

	return coordinate




def read_matrix(input_list, pile_name):
	''' Function to read stiffness matrix for specific pile name 
		and return a big list of stiffness matrix'''
	
	matrix = []
	extract_status = False

	for line in input_list:
		if line[0:10].strip() == pile_name :
			extract_status = True
			#print(line[0:10])

		if extract_status :
			matrix_line = line[17:].strip()
			if matrix_line == "":
				extract_status = False
			else :
				matrix.append(matrix_line.split())	

	return matrix


def matrix_tofloat(input_list):
	''' Convert extracted stiffness matrix to float '''
	
	converted_matrix = []
	for item in input_list:
		converted_matrix.append([float(i) for i in item])
	
	return converted_matrix


def rearrange_matrix(martixname, rownum):
	''' Function to rearrange big matrix into nested matrix '''
	
	how_many_matrix = int(len(martixname)/rownum)
	arranged_matrix = []
	for i in range(how_many_matrix):
		#print(martixname[i*rownum:(i+1)*rownum])
		arranged_matrix.append(martixname[i * rownum:(i + 1) * rownum])
	
	return arranged_matrix

def average_pilewise (matrixname):
	''' Function to average pile stiffness matrix based on pile '''

	averaged_matrix = []

	for i, item in enumerate(matrixname):
		sum_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		 					   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
							   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
							   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
							   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
							   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

		for each in item:
			#print( f'this is important pile_{i}', each[0])
			sum_matrix = sum_matrix + each
			#print( f'this is SUM pile_{i}', sum_matrix[0])


		#averaged_matrix.append(sum_matrix / len(item))
		averaged_matrix.append(sum_matrix / len(item))
		#print(f'This is the averaged {i}' , averaged_matrix)
	return averaged_matrix

def write_output(avgmatrix, headcoord, out_name, pileprefix):
	''' Function to produce output '''

	with open(out_name + ".js" , 'w') as outfile :
		outfile.write('// Averaged linearised stiffness matrix' + '\n')
		outfile.write('// Note : units are not written, should be consistent with FEM unit ' + '\n')
		outfile.write('\n')
		for i, item in enumerate(avgmatrix):
			write_status = False
			for piledata in headcoord:
				if piledata[1] == pileprefix + str(i+1):
					write_status = True
					xcoord = piledata[2]
					ycoord = piledata[3]
					zcoord = piledata[4]

			if write_status :
				outfile.write(f'AvgStiffPile_{i+1} = SupportPoint(Point({xcoord}, {ycoord}, {zcoord}));' + '\n' )
				outfile.write(f'AvgStiffPile_{i+1}.boundary = BoundaryStiffnessMatrix(Stiffness(0), Stiffness(0), Stiffness(0), Stiffness(0), Stiffness(0), Stiffness(0));' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(1 ,1 , {item[0,0]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(1 ,2 , {item[0,1]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(1 ,3 , {item[0,2]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(1 ,4 , {item[0,3]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(1 ,5 , {item[0,4]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(1 ,6 , {item[0,5]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(2 ,2 , {item[1,1]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(2 ,3 , {item[1,2]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(2 ,4 , {item[1,3]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(2 ,5 , {item[1,4]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(2 ,6 , {item[1,5]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(3 ,3 , {item[2,2]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(3 ,4 , {item[2,3]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(3 ,5 , {item[2,4]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(3 ,6 , {item[2,5]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(4 ,4 , {item[3,3]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(4 ,5 , {item[3,4]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(4 ,6 , {item[3,5]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(5 ,5 , {item[4,4]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(5 ,6 , {item[4,5]} );' + '\n')
				outfile.write(f'AvgStiffPile_{i+1}.boundary.setStiffness(6 ,6 , {item[5,5]} );' + '\n')
				outfile.write('\n')

			else:
				outfile.write(f'Pile_{i+1} not found ERROR' + '\n' )

	return 



''' --- execute program ---  '''

#Check if control file is given and valid
resargv = checkargv()

if resargv ==1:
	# echoing the name of the control file
	print('reading control file : ' , sys.argv[1] )
else:
	# spit this when control file not found and stop the program
	sys.exit('ERROR -- Oops! Control file not found')


# Reading the control file 
ctrlpar = to_list(sys.argv[1], True)

print(ctrlpar[0])

#Loop over all the files
combined_matrix = []
for i in range(1, int(ctrlpar[2]) + 1, 1) :
	
	pile_name = ctrlpar[0] + str(i)
	stiffnes_file_name = ctrlpar[1][:-4] + "_" + str(i) + ".LIS"

	#Reading file into list
	raw_list = to_list(stiffnes_file_name, False)

	#Read the matrix for specific pile
	raw_matrix = read_matrix(raw_list, pile_name)

	#Convert matrix to float and to array 
	all_matrix = np.array(matrix_tofloat(raw_matrix))

	#Nested matricess inside a list, grouped per pile, 
	stiffness_matrix = rearrange_matrix(all_matrix, 6)

	#Put all the matricess into a big list 
	combined_matrix.append(stiffness_matrix)

# Perform stiffness averaging - pilewise
avg_matrix = average_pilewise(combined_matrix)

''' CHECK OUTPUT ---------- '''

#print("Length of the combined matrix (should be 16) :", len(combined_matrix))
#print("This is the matrix pile 1 \n ", combined_matrix[0])

'''
sumsum = 0.0
for mat in combined_matrix[0]:
	print(mat[0,0])
	sumsum = sumsum + mat[0,0]

print(sumsum)


#print("This is the length of pile 1 matrix (should be 8): ", len(combined_matrix[0]))
#print("This is the length averaged matrix for all piles (should be 16) : ", len(avg_matrix))
#print("This is the dimension of the Averaged Pile1 matrix (should be 6,6):", avg_matrix[0].shape )


print("This is the matrix pile 3 \n ", combined_matrix[2])
print('\n')
print('\n')
'''
''' CHECK OUTPUT ---------- '''

# Read single file to just extract the pile head coordinates
single_raw_list = to_list(ctrlpar[1][:-4] + "_" + "1.LIS", False)
pilecoord = pile_head_coord(single_raw_list)

#print(*pilecoord, sep='\n')

#Writing output file
output_filename = ctrlpar[3]
if output_filename.strip() == "" :
	print('Please put the output file name in the control file line 4, no output has been produced')
else :
	print('Writing output : ', output_filename + ".js")
	write_output(avg_matrix, pilecoord, output_filename, ctrlpar[0])






















