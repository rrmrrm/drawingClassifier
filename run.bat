::inputSize
set inputSize=400
::expOutputSize
set expOutputSize=6 
::epochs_num
set epochs_num=50
::testDataRatio
set testDataRatio=0.5
::batchSize
set batchSize=18
::inputFileName
set inputFileName=concatenatedCsv/concatenated.csv
::outputFileName
set outputFileName=trainedNetwork.txt
::saveModelPath
set saveModelPath=modelSaveInTfFormat
::loadModelPath
set loadModelPath=modelInTfFormatToRetrain
::retrain
set retrain=0
py drawingClassifier.py %inputSize% %expOutputSize% %epochs_num% %testDataRatio% %batchSize% %inputFileName% %outputFileName% %saveModelPath% %loadModelPath% %retrain% > trainLog.txt
pause