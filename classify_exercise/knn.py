import pandas
import numpy as np 
from numpy import linalg as LA
import operator
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
from __future__ import division
import math

# Load data using pandas
def loadDataset(training_File, test_File):
	#training_File = 'data_train_marbles_classify.csv'
	#test_File = 'data_test_marbles_classify.csv'
	names = ['group', 'id', 'sample_id','quarry', 'd18O', 'd13C', 'dolomite', 'epr_intens', 'epr_linwid', 'colour', 'mgs' ]
	trainingSet = pandas.read_csv(training_File, names = names)
	testSet = pandas.read_csv(test_File, names = names)
	dataSet = trainingSet.append(testSet)
	print (dataSet.shape)

# Calculate Euclidean distance
def euclideanDistance(instance1, instance2, length):
	#distance = np.linalg.norm(instance1 - instance2)
	distance = 0
	for x in range(length):
		distance += pow(instance1[x] - instance2[x], 2)
		distance += np.linalg.norm(instance1[x] - instance2[x])
	return math.sqrt(distance)


# Locate most similar neighbors
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance) - 1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x],dist))

	distances.sort(key = operator.itemgetter(1))
	neighbors = []

	for x in range(k):
		neighbors.append(distances[x][0])

	return neighbors

# Summarize a prediction from neighbors in Python
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1

		else:
			classVotes[response] = 1

	sortedVotes = sorted(classVotes.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedVotes[0][0]


# Calculate accuracy of predictions
def getAccurary(testSet, predictions):
	correct = 0.0
	for x in xrange(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1.0
	return (float(correct)/float(len(testSet))) * 100.0
'''
testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
predictions = ['a', 'a', 'a', 'a']
accuracy = getAccurary(testSet, predictions)
print (accuracy)
'''

def main():
	# Prepare data
	trainingSet = []
	testSet = []
	training_File = 'data_train_marbles_classify.csv'
	test_File = 'data_test_marbles_classify.csv'
	loadDataset(training_File, test_File)

	# Generate predictions
	predictions = []
	k = 6
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

	accuracy = getAccurary(testSet, predictions)
	print ('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
	main()