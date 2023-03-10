"""VICRegANNpt_VICRegANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see VICRegANNpt_main.py

# Usage:
see VICRegANNpt_main.py

# Description:
VICRegANNpt Variance-Invariance-Covariance Regularization artificial neural network (VICRegANN)

"""

import torch as pt
from VICRegANNpt_globalDefs import *
import VICRegANNpt_VICRegANNmodel
import ANNpt_data
import random

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
				
	print("creating new model")
	config = VICRegANNpt_VICRegANNmodel.VICRegANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		hiddenLayerSize = hiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples
	)
	model = VICRegANNpt_VICRegANNmodel.VICRegANNmodel(config)
	return model
	

def generateVICRegANNpairedDatasets(dataset):
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	dataset1 = ANNpt_data.shuffleDataset(dataset)
	dataset2 = ANNpt_data.shuffleDataset(dataset)
	dataset1 = orderDatasetByClassRobust(dataset1, numberOfClasses)
	dataset2 = orderDatasetByClassRobust(dataset2, numberOfClasses)
	shuffleIndices = list(range(0, ANNpt_data.getDatasetSize(dataset)))
	random.shuffle(shuffleIndices)
	dataset1 = dataset1.select(shuffleIndices)
	dataset2 = dataset2.select(shuffleIndices)
	return dataset1, dataset2

def orderDatasetByClassRobust(dataset, numberOfClasses):
	#orders dataset by class but otherwise retains the order of dataset items
	datasetSize = ANNpt_data.getDatasetSize(dataset)
	orderIndicesClassList = [[] for classIndex in range(numberOfClasses)]
	for i in range(datasetSize):
		row = dataset[i]
		target = row[classFieldName]
		orderIndicesClassList[target].append(i)
	orderIndices = [i for orderIndicesClass in orderIndicesClassList for i in orderIndicesClass]	#python flatten list of list
	dataset = dataset.select(orderIndices)
	return dataset
	
def generateVICRegANNpairedBatch(batch1, batch2):
	#print("batch1 = ", batch1)
	#print("batch2 = ", batch2)
	x1 = batch1[0]
	x2 = batch2[0]
	y = batch1[1]
	x1 = pt.unsqueeze(x1, dim=1)
	x2 = pt.unsqueeze(x2, dim=1)
	x = pt.concat([x1, x2], dim=1)
	batch = [x, y]
	return batch
