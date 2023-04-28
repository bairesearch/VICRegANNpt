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


#approximate VICRegANNtf parameters; n_h =  [5, 15, 9, 3]
batchSize = 64	#100
numberOfLayers = 4
hiddenLayerSize = 10	

usePairedDataset = True	#required
trainLocal = True	#required	#local learning rule	#disable for debug (standard backprop algorithm)
debugOnlyTrainLastLayer = False
lambdaHyperparameter = 1.0 #invariance coefficient	#base condition > 1
muHyperparameter = 1.0	#invariance coefficient	#base condition > 1
nuHyperparameter = 1.0 #covariance loss coefficient	#set to 1
optimiserAdam = False	#CHECKTHIS
if(trainLocal):
	trainGreedy = True	 #optional	#train layers with all data consecutively	#default tf implementation
useCustomWeightInitialisation = True	#emulate VICRegANNtf
useCustomBiasInitialisation = True	#emulate VICRegANNtf	#initialise biases to zero

debugDataNormalisation = False
debugParameterInitialisation = False
debugVICRegLoss = False
if(debugDataNormalisation or debugParameterInitialisation or debugVICRegLoss):
	debugSmallBatchSize = True

workingDrive = '/large/source/ANNpython/VICRegANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelVICRegANN'

usePositiveWeights = True	#experimental
if(usePositiveWeights):
	usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required
	#activationFunctionType = "softmax"
	activationFunctionType = "none"
	normaliseActivationSparsity = True
	debugUsePositiveWeightsVerify = False


