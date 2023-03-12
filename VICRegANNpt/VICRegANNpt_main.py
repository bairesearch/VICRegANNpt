"""VICRegANNpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics

# Usage:
source activate pytorchsenv
python VICRegANNpt_main.py

# Description:
VICRegANNpt main - learning algorithm experiment (LRE) artificial neural network

"""

import torch
from tqdm.auto import tqdm
from torch import optim

from VICRegANNpt_globalDefs import *
if(useAlgorithmVICRegANN):
	import VICRegANNpt_VICRegANN
if(usePositiveWeights):
	import ANNpt_linearSublayers
import ANNpt_data

#https://huggingface.co/docs/datasets/tabular_load

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
	dataset = ANNpt_data.loadDataset()
	if(stateTrainDataset):
		if(useAlgorithmVICRegANN):
			model = VICRegANNpt_VICRegANN.createModel(dataset['train'])	#dataset['test'] not possible as test does not contain all classes
		processDataset(True, dataset['train'], model)
	if(stateTestDataset):
		model = loadModel()
		processDataset(False, dataset['test'], model)

def createOptimizer():
	if(optimiserAdam):
		optim = torch.optim.Adam(model.parameters(), lr=learningRate)
	else:
		optim = torch.optim.SGD(model.parameters(), lr=learningRate)
	return optim
	
def processDataset(trainOrTest, dataset, model):

	if(trainOrTest):
		if(trainLocal):
			optim = [None]*model.config.numberOfLayers
			for layerIndex in range(model.config.numberOfLayers):
				optimLayer = torch.optim.Adam(model.parameters(), lr=learningRate)
				optim[layerIndex] = optimLayer
		else:
			optim = torch.optim.Adam(model.parameters(), lr=learningRate)
		model.to(device)
		model.train()	
		numberOfEpochs = trainNumberOfEpochs
	else:
		model.to(device)
		model.eval()
		numberOfEpochs = 1
		
	for epoch in range(numberOfEpochs):
		dataset1, dataset2 = VICRegANNpt_VICRegANN.generateVICRegANNpairedDatasets(dataset)
		
		if(trainGreedy):
			maxLayer = model.config.numberOfLayers
		else:
			maxLayer = 1
		for l in range(maxLayer):
			if(trainGreedy):
				print("trainGreedy: l = ", l)
			
			if(printAccuracyRunningAverage):
				(runningLoss, runningAccuracy) = (0.0, 0.0)

			if(dataloaderRepeatLoop):
				numberOfDataloaderIterations = dataloaderRepeatSize
			else:
				numberOfDataloaderIterations = 1
			for dataLoaderIteration in range(numberOfDataloaderIterations):
	
				loader1 = ANNpt_data.createDataLoaderTabularPaired(dataset1, dataset2)	#required to reset dataloader and still support tqdm modification
				loop1 = tqdm(loader1, leave=True)
				for batchIndex, batch in enumerate(loop1):

					if(trainOrTest):
						loss, accuracy = trainBatch(batchIndex, batch, model, optim, l)
					else:
						loss, accuracy = testBatch(batchIndex, batch, model, l)

					if(printAccuracyRunningAverage):
						(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))

					loop1.set_description(f'Epoch {epoch}')
					loop1.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		saveModel(model)
					
def trainBatch(batchIndex, batch, model, optim, l=None):
	if(not trainLocal):
		optim.zero_grad()
	loss, accuracy = propagate(True, batchIndex, batch, model, optim, l)
	if(not trainLocal):
		loss.backward()
		optim.step()
	
	if(usePositiveWeights):
		if(usePositiveWeightsClampModel):
			ANNpt_linearSublayers.weightsSetPositiveModel(model)

	if(batchIndex % modelSaveNumberOfBatches == 0):
		saveModel(model)
	loss = loss.item()
			
	return loss, accuracy
			
def testBatch(batchIndex, batch, model, l=None):

	loss, accuracy = propagate(False, batchIndex, batch, model, l)

	loss = loss.detach().cpu().numpy()
	
	return loss, accuracy

def saveModel(model):
	torch.save(model, modelPathNameFull)

def loadModel():
	print("loading existing model")
	model = torch.load(modelPathNameFull)
	return model
		
def propagate(trainOrTest, batchIndex, batch, model, optim=None, l=None):
	(x, y) = batch
	y = y.long()
	x = x.to(device)
	y = y.to(device)
	if(debugDataNormalisation):
		print("x = ", x)
		print("y = ", y)
		
	loss, accuracy = model(trainOrTest, x, y, optim, l)
	return loss, accuracy
				
if(__name__ == '__main__'):
	main()






