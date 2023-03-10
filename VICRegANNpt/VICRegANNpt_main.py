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

def processDataset(trainOrTest, dataset, model):

	if(trainOrTest):
		model.to(device)
		if(trainLocal):
			optim = [None]*model.config.numberOfLayers
			for layerIndex in range(model.config.numberOfLayers):
				optimLayer = torch.optim.Adam(model.parameters(), lr=learningRate)
				optim[layerIndex] = optimLayer
		else:
			optim = torch.optim.Adam(model.parameters(), lr=learningRate)
		if(not trainLocal):
			model.train()
		else:
			model.eval()		
		numberOfEpochs = trainNumberOfEpochs
	else:
		model.to(device)
		model.eval()
		numberOfEpochs = 1
		
	for epoch in range(numberOfEpochs):
		dataset1, dataset2 = VICRegANNpt_VICRegANN.generateVICRegANNpairedDatasets(dataset)

		loader1 = ANNpt_data.createDataLoader(dataset1)	#required to reset dataloader and still support tqdm modification
		loop1 = tqdm(loader1, leave=True)
		loader2 = ANNpt_data.createDataLoader(dataset2)
		loader2iter = iter(loader2)
				
		if(printAccuracyRunningAverage):
			(runningLoss, runningAccuracy) = (0.0, 0.0)
		
		for batchIndex, batch1 in enumerate(loop1):
			batch2 = next(loader2iter)
			batch = VICRegANNpt_VICRegANN.generateVICRegANNpairedBatch(batch1, batch2)
			
			if(trainOrTest):
				loss, accuracy = trainBatch(batchIndex, batch, model, optim)
			else:
				loss, accuracy = testBatch(batchIndex, batch, model)
			
			if(printAccuracyRunningAverage):
				(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))
				
			loop1.set_description(f'Epoch {epoch}')
			loop1.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		saveModel(model)
					
def trainBatch(batchIndex, batch, model, optim):
	if(not trainLocal):
		optim.zero_grad()
	loss, accuracy = propagate(True, batchIndex, batch, model, optim)
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
			
def testBatch(batchIndex, batch, model):

	loss, accuracy = propagate(False, batchIndex, batch, model)

	loss = loss.detach().cpu().numpy()
	
	return loss, accuracy

def saveModel(model):
	torch.save(model, modelPathNameFull)

def loadModel():
	print("loading existing model")
	model = torch.load(modelPathNameFull)
	return model
		
def propagate(trainOrTest, batchIndex, batch, model, optim=None):
	(x, y) = batch
	y = y.long()
	x = x.to(device)
	y = y.to(device)
	#print("x = ", x)
	#print("y = ", y)
	loss, accuracy = model(trainOrTest, x, y, optim)
	return loss, accuracy
				
if(__name__ == '__main__'):
	main()






