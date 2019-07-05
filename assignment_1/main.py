############################################################################################
#																					       #
#								Author: Victor Mendoza						       	       #
#								Course: Machine Learning			                	   #
#          						Assignment 1 - Decision Tree & WEKA						   #
#											                                               #
############################################################################################

import numpy as np

############################################################################################
#											                                               #
# Helpers functions to calculate Entropy and Gain 										   #
# We should study zoo.xls file without using computational resources                       #
# Build a decision tree to classify animals into seven different classes                   #
#											                                               #
# Classes:									                                               #
#											                                               #
#	1 - Mammal								                                               #
#	2 - Bird								                                               #
#	3 - Reptile								                                               #
#	4 - Fish								                                               #
#	5 - Amphibian							                                               #
#	6 - Insect								                                               #
#	7 - Invertebrate						                                               #
#											                                               #
############################################################################################

def entropy(S):
	"""
	S -> numpy.array

	return
	entropy -> entropy value for an attribute
		based on formula:

			sum(-p_i * log_2 p_i)

			Where p_i is a proportion on match in the set S
	"""
	n = np.sum(S) # Calculate the total # of samples
	S = S/n # Obtain proportion p for each element in S
	return np.sum(-S*np.log2(S)) # Calcute entropy os S


def gain(S,A,possible_values=2):
	E_S = entropy(S)
	n_S = np.sum(S)
	if possible_values==2:
		E_A_0 = entropy(A[0])
		E_A_1 = entropy(A[1])

		n_A_0 = np.sum(A[0])
		n_A_1 = np.sum(A[1])

		return E_S - n_A_0/n_S*E_A_0 - n_A_1/n_S*E_A_1
############################################################################################
# 											                                               #
# Study of Entropy and gain					                                               #
# 											                                               #
############################################################################################

S = np.array([41,20,5,13,4,8,10]) # Samples per class
E_S = entropy(S) # Entropy os S
n_S = np.sum(S) # Total values for S

print("\nTotal Entropy: {0:.3f}".format(E_S))

print("\nCalculating best node for first level:\n")

# Attribute 2 (Value 0 or 1)

A2 = np.array([
		[39,4], # Samples for Attribute 2 where values are 1
		[2,20,5,13,4,4,10] # Samples for Attribute 2 where values are 0
	])

print("\tGain for Attribute 2: {0:.3f}".format(
		gain(S,A2)
	))

# Attribute 3 (Value 0 or 1)

A3 = np.array([
		[20], # Samples for Attribute 3 where values are 1
		[41,5,13,4,8,10] # Samples for Attribute 3 where values are 0
	])

print("\tGain for Attribute 3: {0:.3f}".format(
		gain(S,A3)
	))

# Attribute 4 (Value 0 or 1)

A4 = np.array([
		[1,20,4,13,4,8,9], # Samples for Attribute 4 where values are 1
		[40,1,1] # Samples for Attribute 4 where values are 0
	])

print("\tGain for Attribute 4: {0:.3f}".format(
		gain(S,A4)
	))

# Attribute 5 (Value 0 or 1)

A5 = np.array([
		[41], # Samples for Attribute 5 where values are 1
		[20,5,13,4,8,10] # Samples for Attribute 5 where values are 0
	])

print("\tGain for Attribute 5: {0:.3f}".format(
		gain(S,A5)
	))

# Attribute 6 (Value 0 or 1)

A6 = np.array([
		[2,16,6], # Samples for Attribute 6 where values are 1
		[39,4,5,13,4,2,10] # Samples for Attribute 6 where values are 0
	])

print("\tGain for Attribute 6: {0:.3f}".format(
		gain(S,A6)
	))

# Attribute 7 (Value 0 or 1)

A7 = np.array([
		[5,6,1,13,4,6], # Samples for Attribute 7 where values are 1
		[36,14,4,8,4] # Samples for Attribute 7 where values are 0
	])

print("\tGain for Attribute 7: {0:.3f}".format(
		gain(S,A7)
	))

# Attribute 8 (Value 0 or 1)

A8 = np.array([
		[22,9,4,9,3,1,8], # Samples for Attribute 8 where values are 1
		[19,11,1,4,1,7,2] # Samples for Attribute 8 where values are 0
	])

print("\tGain for Attribute 8: {0:.3f}".format(
		gain(S,A8)
	))

# Attribute 9 (Value 0 or 1)

A9 = np.array([
		[40,4,13,4], # Samples for Attribute 9 where values are 1
		[1,20,1,8,10] # Samples for Attribute 9 where values are 0
	])

print("\tGain for Attribute 9: {0:.3f}".format(
		gain(S,A9)
	))

# Attribute 10 (Value 0 or 1)

A10 = np.array([
		[41,20,5,13,4], # Samples for Attribute 10 where values are 1
		[8,10] # Samples for Attribute 10 where values are 0
	])

print("\tGain for Attribute 10: {0:.3f}".format(
		gain(S,A10)
	))

# Attribute 11 (Value 0 or 1)

A11 = np.array([
		[41,20,4,4,8,3], # Samples for Attribute 11 where values are 1
		[1,13,7] # Samples for Attribute 11 where values are 0
	])

print("\tGain for Attribute 11: {0:.3f}".format(
		gain(S,A11)
	))

# Attribute 12 (Value 0 or 1)

A12 = np.array([
		[2,1,1,2,1], # Samples for Attribute 12 where values are 1
		[41,20,3,12,3,6,9] # Samples for Attribute 12 where values are 0
	])

print("\tGain for Attribute 12: {0:.3f}".format(
		gain(S,A12)
	))

# Attribute 13 (Value 0 or 1)

A13 = np.array([
		[4,13], # Samples for Attribute 13 where values are 1
		[37,20,5,4,8,10] # Samples for Attribute 13 where values are 0
	])

print("\tGain for Attribute 13: {0:.3f}".format(
		gain(S,A13)
	))

# Attribute 14 (Value 0 or 1)

A14 = np.array([
		[38,20,2,4,7,6], # Samples for Attribute 14 where values are 1
		[3,3,13,1,4] # Samples for Attribute 14 where values are 0
	])

print("\tGain for Attribute 14: {0:.3f}".format(
		gain(S,A14)
	))

# Attribute 15 (Value 0 or 1)

A15 = np.array([
		[32,20,5,13,1,1], # Samples for Attribute 15 where values are 1
		[9,3,8,9] # Samples for Attribute 15 where values are 0
	])

print("\tGain for Attribute 15: {0:.3f}".format(
		gain(S,A15)
	))

# Attribute 16 (Value 0 or 1)

A16 = np.array([
		[7,3,1,1], # Samples for Attribute 16 where values are 1
		[34,17,5,12,4,7,10] # Samples for Attribute 16 where values are 0
	])

print("\tGain for Attribute 16: {0:.3f}".format(
		gain(S,A16)
	))

# Attribute 17 (Value 0 or 1)

A17 = np.array([
		[32,6,1,4,1], # Samples for Attribute 17 where values are 1
		[9,14,4,8,4,8,9] # Samples for Attribute 17 where values are 0
	])

print("\tGain for Attribute 17: {0:.3f}".format(
		gain(S,A17)
	))