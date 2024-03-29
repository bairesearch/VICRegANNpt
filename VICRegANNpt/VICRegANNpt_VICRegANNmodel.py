"""VICRegANNpt_VICRegANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

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

		self.layersLinear, self.layersActivation = self.generateNetworkLayers(config, False)
		if(trainAutoencoder):
			self.layersLinearReverse, self.layersActivationReverse = self.generateNetworkLayers(config, True)
			self.lossFunctionReverse = nn.MSELoss()

		if(networkHemispherical):
			self.layersLinear2, self.layersActivation2 = self.generateNetworkLayers(config, False)
			if(trainAutoencoder):
				self.layersLinearReverse2, self.layersActivationReverse2 = self.generateNetworkLayers(config, True)

		self.lossFunction = nn.CrossEntropyLoss()
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		ANNpt_linearSublayers.weightsSetPositiveModel(self)

	def generateNetworkLayers(self, config, reverse):
		layersLinearList = []
		layersActivationList = []
		for layerIndex in range(config.numberOfLayers):
			linear = ANNpt_linearSublayers.generateLinearLayer(self, layerIndex, config, reverse=reverse)
			layersLinearList.append(linear)
		for layerIndex in range(config.numberOfLayers):
			activation = ANNpt_linearSublayers.generateActivationLayer(self, layerIndex, config, reverse=reverse)
			layersActivationList.append(activation)
		layersLinear = nn.ModuleList(layersLinearList)
		layersActivation = nn.ModuleList(layersActivationList)
		return layersLinear, layersActivation
	
	def forward(self, trainOrTest, x, y, optim, l=None):	
		if(trainVicreg and trainOrTest and not debugOnlyTrainLastLayer):
			loss, accuracy = self.forwardBatchVICReg(x, y, optim, l)
		else:
			if(trainVicreg and trainOrTest and debugOnlyTrainLastLayer):
				layerIndexLast = self.config.numberOfLayers-1
				opt = optim[layerIndexLast]
				opt.zero_grad()
			loss, accuracy = self.forwardBatchStandard(x, y)	#standard backpropagation
			if(trainVicreg and trainOrTest and debugOnlyTrainLastLayer):
				loss.backward()
				opt.step()
				
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
				if(trainLocal):
					x1, x2, loss, accuracy = self.trainLayer(layerIndex, x1, x2, y, optim, True)
				else:
					train = False
					if((layerIndex == maxLayer-2) or (layerIndex == maxLayer-1)):
						train = True	#second last (final hidden/backbone) layer or final output layer
					x1, x2, loss, accuracy = self.trainLayer(layerIndex, x1, x2, y, optim, train)
					#final hidden layer/backbone (will backpropagate vicreg loss)
		return loss, accuracy

	def trainLayer(self, layerIndex, x1, x2, y, optim, train):

		loss = None
		accuracy = 0.0
		if(train):
			opt = optim[layerIndex]
			opt.zero_grad()

		if(layerIndex == self.config.numberOfLayers-1):
			loss, accuracy = self.forwardLayerLast(layerIndex, x1, y)
		else:
			x1, x2, loss = self.forwardLayerVICReg(layerIndex, x1, x2)
			
		if(train):
			loss.backward()
			opt.step()

		return x1, x2, loss, accuracy
		
	def forwardLayerVICReg(self, layerIndex, x1, x2):
		if(trainLocal):
			x1 = x1.detach()
			x2 = x2.detach()

		a1, z1 = self.propagatePairElementLayer(0, layerIndex, x1)
		a2, z2 = self.propagatePairElementLayer(1, layerIndex, x2)
		lossVICreg = VICRegANNpt_VICRegANNloss.calculatePropagationLossVICRegANN(a1, a2)
	
		if(trainAutoencoder):
			a1AE, z1AE = self.propagatePairElementLayerReverse(0, layerIndex, a1)
			a2AE, z2AE = self.propagatePairElementLayerReverse(1, layerIndex, a2)
			lossAE1 = self.lossFunctionReverse(a1AE, x1)
			lossAE2 = self.lossFunctionReverse(a2AE, x2)
			loss = (lossVICreg + (lossAE1+lossAE2)/2)/2
		else:
			loss = lossVICreg
		
		if(debugParameterInitialisation):
			print("z1 = ", z1)
			print("z2 = ", z2)
			EW = self.layersLinear[layerIndex].weight
			print("EW = ", EW)
		
		return a1, a2, loss
	
	def propagatePairElementLayer(self, pairIndex, layerIndex, x):
		if(networkHemispherical and (pairIndex == 1)):
			z = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear2[layerIndex])
			a = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, z, self.layersActivation2[layerIndex])
		else:
			z = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
			a = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, z, self.layersActivation[layerIndex])
		return a, z
		
	def propagatePairElementLayerReverse(self, pairIndex, layerIndex, x):
		if(networkHemispherical and (pairIndex == 1)):
			z = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinearReverse2[layerIndex])
			a = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, z, self.layersActivationReverse2[layerIndex])
		else:
			z = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinearReverse[layerIndex])
			a = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, z, self.layersActivationReverse[layerIndex])
		return a, z

	def forwardLayerLast(self, layerIndex, x, y):
		x = x.detach()	#trainVicreg trains last layer using backprop only
		x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		return loss, accuracy


