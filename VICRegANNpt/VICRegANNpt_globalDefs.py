"""VICRegANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
VICRegANNpt globalDefs

"""

trainVicreg = True
vicregBiologicalMods = True

#initialise (dependent vars);
partiallyAlignLayer = False
vicregSimilarityLossOnly = False
trainMostAlignedNeurons = False

if(trainVicreg):
	if(vicregBiologicalMods):
		trainLocal = True	#local learning rule	#required
		if(trainLocal):
			trainGreedy = False	 #optional	#train layers with all data consecutively	#default tf implementation
		trainAutoencoder = False
		if(trainAutoencoder):
			vicregSimilarityLossOnly = False
		networkHemispherical = True	#optional	#propagate through two paired networks
		if(networkHemispherical):
			partiallyAlignLayer = True	#experimental; align a fraction of layer neurons
			vicregSimilarityLossOnly = False	#experimental; minor connectivity differences between paired network architectures might add regularisation
			trainMostAlignedNeurons = False	#experimental; only train already partially aligned neurons
			if(partiallyAlignLayer):
				partiallyAlignLayerFraction = 0.5	#fraction of layer neurons to align
				partiallyAlignLayerIgnoreValue = 1.0	#set arbitrary activation value during training to ignore alignment
			#networkHemisphericalStereoInput = True	#TODO	#use stereo input (vision, audio etc) - do not select matched/ablated input between network pairs
			#networkHemisphericalAlignment = 0.1	#TODO	#degree of hemispherical alignment - fraction of neurons per layer to align (masked)
			if(trainMostAlignedNeurons):
				#trainMostAlignedNeuronsMethod = "softmax"
				#trainMostAlignedNeuronsMethod = "thresholded"
				#trainMostAlignedNeuronsThresholdMin = 1.5
				trainMostAlignedNeuronsMethod = "topk"	#incomplete
				trainMostAlignedNeuronsTopK = 5	#number of neurons per layer to train (topk)
			sparseLinearLayers = True	#add minor connectivity differences between paired network architectures
			if(sparseLinearLayers):
				sparseLinearLayersLevel = 0.8	#0.5	#fraction of non-zeroed connections
	else:
		trainLocal = False	#non-local (final hidden/backbone layer) vicreg training

#approximate VICRegANNtf parameters; n_h =  [5, 15, 9, 3]
batchSize = 64	#100
numberOfLayers = 4
if(partiallyAlignLayer):
	hiddenLayerSize = 100
elif(trainMostAlignedNeurons):
	hiddenLayerSize = 1000
else:
	hiddenLayerSize = 10

usePairedDataset = True	#required
lambdaHyperparameter = 1.0 #invariance coefficient	#base condition > 1
muHyperparameter = 1.0	#invariance coefficient	#base condition > 1
nuHyperparameter = 1.0 #covariance loss coefficient	#set to 1
optimiserAdam = False	#CHECKTHIS
useCustomWeightInitialisation = True	#emulate VICRegANNtf
useCustomBiasInitialisation = True	#emulate VICRegANNtf	#initialise biases to zero

if(vicregBiologicalMods):
	usePositiveWeights = True
	if(usePositiveWeights):
		usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required
		#activationFunctionType = "softmax"
		activationFunctionType = "none"
		normaliseActivationSparsity = True
		debugUsePositiveWeightsVerify = False

debugDataNormalisation = False
debugParameterInitialisation = False
debugVICRegLoss = False
if(debugDataNormalisation or debugParameterInitialisation or debugVICRegLoss):
	debugSmallBatchSize = True
debugOnlyTrainLastLayer = False
if(debugOnlyTrainLastLayer):
	hiddenLayerSize = 100	#temporarily emulate partiallyAlignLayer

workingDrive = '/large/source/ANNpython/VICRegANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelVICRegANN'

