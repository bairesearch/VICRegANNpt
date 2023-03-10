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
from VICRegANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers
import VICRegANNpt_VICRegANNloss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	

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

	def forward(self, trainOrTest, x, y, optim):	
		if(trainLocal and trainOrTest and not debugOnlyTrainLastLayer):
			loss, accuracy = self.forwardBatchVICReg(x, y, optim)
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
		
	def forwardBatchVICReg(self, x, y, optim):
		x1 = x[:, 0]
		x2 = x[:, 1]
		#lossAverage = 0.0
		for layerIndex in range(self.config.numberOfLayers):
			if(layerIndex == self.config.numberOfLayers-1):
				loss, accuracy = self.forwardLastLayer(layerIndex, x1, y)	#only calculate loss/accuracy for first experience in matched class pair
			else:
				x1, x2, loss = self.trainLayerVICReg(layerIndex, x1, x2, optim)
			#lossAverage = lossAverage + loss
		#lossAverage = lossAverage/batchSize
		return loss, accuracy

	def trainLayerVICReg(self, layerIndex, x1, x2, optim):
		x1 = x1.detach()
		x2 = x2.detach()
		
		optim = optim[layerIndex]
		optim.zero_grad()

		x1 = self.propagatePairElementLayer(layerIndex, x1)
		x2 = self.propagatePairElementLayer(layerIndex, x2)
		loss = VICRegANNpt_VICRegANNloss.calculatePropagationLossVICRegANN(x1, x2)
			
		loss.backward()
		optim.step()

		return x1, x2, loss
	
	def propagatePairElementLayer(self, layerIndex, x):
		x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
		x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex])
		return x

	def forwardLastLayer(self, layerIndex, x, y):
		x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		return loss, accuracy


