"""VICRegANNpt_VICRegANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see VICRegANNpt_main.py

# Usage:
see VICRegANNpt_main.py

# Description:
VICRegANNpt Variance-Invariance-Covariance Regularization artificial neural network (VICRegANN) model

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers
import VICRegANNpt_VICRegANNloss
	

class VICRegANNconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples

class VICRegANNmodel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		layersLinearList = []
		layersActivationList = []
		for layerIndex in range(config.numberOfLayers):
			linear = ANNpt_linearSublayers.generateLinearLayer(self, layerIndex, config)
			layersLinearList.append(linear)
		for layerIndex in range(config.numberOfLayers):
			activation = ANNpt_linearSublayers.generateActivationLayer(self, layerIndex, config)
			layersActivationList.append(activation)
		self.layersLinear = nn.ModuleList(layersLinearList)
		self.layersActivation = nn.ModuleList(layersActivationList)
	
		self.lossFunction = nn.CrossEntropyLoss()
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		ANNpt_linearSublayers.weightsSetPositiveModel(self)

		if(trainLocal):
			self.previousSampleStatesLayerList = [None]*config.numberOfLayers
			self.previousSampleClass = None

	def forward(self, trainOrTest, x, y, optim, l=None):	
		if(trainLocal and trainOrTest and not debugOnlyTrainLastLayer):
			loss, accuracy = self.forwardBatchVICReg(x, y, optim, l)
		else:
			loss, accuracy = self.forwardBatchStandard(x, y)	#standard backpropagation
			
		return loss, accuracy

	def forwardBatchStandard(self, x, y):
		#print("forwardBatchStandard")
		x = x[:, 0]	#only optimise final layer weights for first experience in matched class pair
		for layerIndex in range(self.config.numberOfLayers):
			if(debugOnlyTrainLastLayer and (layerIndex == self.config.numberOfLayers-1)):
				x = x.detach()
			x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
			if(layerIndex != self.config.numberOfLayers-1):
				x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex])
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		return loss, accuracy
		
	def forwardBatchVICReg(self, x, y, optim, l=None):
		if(trainGreedy):
			maxLayer = l+1
		else:
			maxLayer = self.config.numberOfLayers
		x1 = x[:, 0]
		x2 = x[:, 1]
		accuracy = 0.0	#in case accuracy calculations are not possible
		for layerIndex in range(maxLayer):
			if(trainGreedy):
				x1, x2, loss, accuracy = self.trainLayer(layerIndex, x1, x2, y, optim, (layerIndex == l))
			else:
				x1, x2, loss, accuracy = self.trainLayer(layerIndex, x1, x2, y, optim, True)
		return loss, accuracy

	def trainLayer(self, layerIndex, x1, x2, y, optim, train):

		loss = None
		accuracy = 0.0
		if(train):
			optim = optim[layerIndex]
			optim.zero_grad()

		if(layerIndex == self.config.numberOfLayers-1):
			loss, accuracy = self.forwardLayerLast(layerIndex, x1, y)
		else:
			x1, x2, loss = self.forwardLayerVICReg(layerIndex, x1, x2)
			
		if(train):
			loss.backward()
			optim.step()

		return x1, x2, loss, accuracy
		
	def forwardLayerVICReg(self, layerIndex, x1, x2):
		x1 = x1.detach()
		x2 = x2.detach()

		x1, z1 = self.propagatePairElementLayer(layerIndex, x1)
		x2, z2 = self.propagatePairElementLayer(layerIndex, x2)

		if(debugParameterInitialisation):
			print("z1 = ", z1)
			print("z2 = ", z2)
			EW = self.layersLinear[layerIndex].weight
			print("EW = ", EW)
			
		loss = VICRegANNpt_VICRegANNloss.calculatePropagationLossVICRegANN(x1, x2)
		
		return x1, x2, loss
	
	def propagatePairElementLayer(self, layerIndex, x):
		z = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
		a = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, z, self.layersActivation[layerIndex])
		return a, z

	def forwardLayerLast(self, layerIndex, x, y):
		x = x.detach()
		x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		return loss, accuracy


