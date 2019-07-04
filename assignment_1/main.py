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

############################################################################################
# 											                                               #
# Entropy for all data						                                               #
# New tree level based on Gain for attribute                                               #
# 											                                               #
############################################################################################