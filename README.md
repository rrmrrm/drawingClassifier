# drawingClassifier
showcase for my simple tensorflow classifier.

doesn't contain the program that creates the learning data and uses the trained neuralnetwork to classify new drawings

drawingClassifier.py:
first:
	if 'retrain' is True, the program loads a previously saved model from 'loadModelPath', removes a few layers from the end of the neural newtwork, and adds a new layers to it. Then it trains the model with the .csv dataset loaded from 'inputFileName'. Only the newly added layers' weights will be trained. 
	Otherwise the program creates a new simple model, and trains the model with the .csv dataset loaded from 'inputFileName'. 
then the program saves the trained model in a tensorflow specific format to 'saveModelPath' directory, 
and exports it in a readable format to 'outputFileName'

arguments for running the file:

inputSize
expOutputSize
epochs_num
testDataRatio
batchSize
inputFileName
outputFileName
saveModelPath
loadModelPath
retrain