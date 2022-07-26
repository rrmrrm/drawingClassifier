from array import array
from math import fabs
import sys
import os
from unicodedata import name

from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.core import Dropout

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
#for debugging:
#inputSize=400
#expOutputSize=6 
#epochs_num=50
#testDataRatio=0.5
#batchSize=18
#inputFileName="concatenatedCsv/concatenated.csv"
#outputFileName="trainedNetwork.txt"
#saveModelPath="modelSaveInTfFormat"
#loadModelPath="modelInTfFormatToRetrain"
#retrain=0

inputSize = int(sys.argv[1])
expOutputSize = int(sys.argv[2])
epochs_num = int(sys.argv[3])
testDataRatio = float(sys.argv[4])
batchSize= int(sys.argv[5])
if len(sys.argv) > 6:
    inputFileName = sys.argv[6]
if len(sys.argv) > 7:
    outputFileName = sys.argv[7]
if len(sys.argv) > 8:
    saveModelPath = sys.argv[8]
if len(sys.argv) > 9:
    loadModelPath = sys.argv[10]
if len(sys.argv) > 10:
    if sys.argv[10] != "0":
        retrain = True
    else:
        retrain = False

# if freezeKeptWights and retraining is True, the weights from the loaded model will be trained as well as the additional weights
freezeKeptWights = True


train = pd.DataFrame();
useTestData = False
if useTestData:
    retrain = True
    inputSize = 2
    expOutputSize = 2
    testDataRatio = 0
    batchSize= 1
    outputFileName = "trainedNet.txt"
    train = pd.DataFrame(
        [
            (0,0, 0,1),
            (1,0, 0,1),
            (0,1, 1,0),
            (1,1, 1,0),
        ],
        columns= ('i0','i1','out_0','out_1')
    )
else:
    #read dataset from .csv file:
    np.set_printoptions(precision=5, suppress=True)
    names_array = np.empty(0)
    for i in range(0, inputSize):
        names_array = np.append(names_array, "cell_"+str(i))
    for i in range(0, expOutputSize):
        names_array = np.append(names_array, "out_"+str(i))
    print(names_array)
    #load csv into 'train' DataFrame
    train = pd.read_csv(
        inputFileName, header=None, names=names_array)

print(train.head())
features = train.copy()

#megkeverem az adatokat(ez a tanítást segíti, és enélkül a test adatok tömbjébe nem feltétlen jutna mindegyik osztályhoz-hoz tartozó mintából):
#forrás: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
print("shuffling data")
features = features.sample(frac=1).reset_index(drop=True)

print("features after shuffling:")
print(features)
#labels csak az elvárt outputokat fogja tartalmazni, features pedig a bemeneteket:
labels = []
#labels = np.delete(labels, 0)
#a labels-be kerülnek a features-ből az elvárt kimenetekhez tartozó oszlopok:
for i in range(0, expOutputSize):
    outputCol = features.pop("out_"+str(i))
    labels.append(outputCol)
labels = np.array(labels)
#az append függvény a feature oszlopait töltötte bele a labels-be, ezért a labels-t transzponálni kell:
labels = np.transpose(labels)
features = np.array(features)

print("features after label separation:")
print(features)
print("labels after separation:")
print(labels)

#kettébontom az adathalmazt teszt és tanító mintákra testDataRation arányban:
sliceAtTrain = int( len(features)*testDataRatio )
#sliceAtTest = int( len(features)*testDataRatio )
featuresTrain = []
labelsTrain = []
featuresTest= []
labelsTest = []
#sliceAtTrain = ö külön eset, mert tomb[:-0] ures tombot jelent:
if sliceAtTrain == 0:
    featuresTrain = features
    labelsTrain = labels
    featuresTest= []
    labelsTest = []
else:
    featuresTrain = features[ : -sliceAtTrain]
    labelsTrain = labels[ : -sliceAtTrain ]
    featuresTest= features[-sliceAtTrain : ]
    labelsTest = labels[-sliceAtTrain : ]

print("trainingData size:", len(featuresTrain))
print("testData size:", len(featuresTest))

models = []
if retrain:
    print("loading model from " + loadModelPath)
    modeltoRetrain = tf.keras.models.load_model(loadModelPath)
    
    layersList = [l for l in modeltoRetrain.layers]
    layersToKeep = len(layersList) -1
    assert(layersToKeep > 0)
    assert(layersToKeep <= len(layersList))
    
    # layerSizesToReplaceRest: az új létrehozandó rétegek méretei.
    # Az utolsó elem felül lesz írva, expOutputSize-ra, így az nem fogja meghatározni semelyik réteg méretét,
    #  de layerSizesToReplaceRest minden egyes eleme egy új réteget jelent
    layerSizesToReplaceRest = [20,20,2]
    assert(len(layerSizesToReplaceRest) > 0)
    
    # opcionális: a meghagyott rétegek tanítását megakadályozom:
    if freezeKeptWights:
        print("freezing loaded weights before retraining.")
        for i in range(0, layersToKeep):
            layersList[i].trainable = False
    else:
        print("un-freezing loaded weights before retraining.")
        for i in range(0, layersToKeep):
            layersList[i].trainable = True
    modeltoRetrain.summary()

    # a betöltött háló rétegeinek méretét egy listában tárolom, az elemei határozzák majd meg az új rétegek számát és méretét:
    layersOutputShapes = [l.output_shape for l in layersList]
    #az inputlayer speciális:
    layersOutputSizes = [inputSize]
    # a többi réteg méretét is megadjuk:
    for i in range(1, layersToKeep):
        layersOutputSizes.append(layersOutputShapes[i][1])
    print("a meghagyott rétegek méretei: ")
    print(layersOutputSizes)

    # itt adom hozzá layersOutputSizes-hez az új létrehozandó rétegek méretét.
    # Az utolsó elem felül lesz írva így az nem fogja meghatározni semelyik réteg méretét,
    #  de minden egyes hozzáadott elem egy új réteget jelent.(ld. lejjebb)
    for s in layerSizesToReplaceRest:
        layersOutputSizes.append(s)

    # az utolsó réteg perceptronjainak a számát frissítjük, hogy a háló képes legyen a megfelelő számú alakzat megkülönböztetésére:
    layersOutputSizes[len(layersOutputSizes)-1] = expOutputSize
    print("new layersOutputSizes: ")
    print(layersOutputSizes)
    # az utolsó meghagyott réteg output-ját átadjuk az első új rétegnek:
    x = modeltoRetrain.layers[layersToKeep-1].output
    for i in range(layersToKeep, len(layersOutputSizes)):
        print(layersOutputSizes[i])
        x = layers.Dense(units=layersOutputSizes[i], activation='sigmoid', name="dense_"+str(i)+"replaced")(x)

    result_model = tf.keras.Model(inputs=layersList[0].input, outputs=x)
    
    result_model.summary()
    
    models = [result_model]
else:
    print("creating model")
    if useTestData:
        # teszteléshez kisebb belső rétegeket hozok létre:
        inputs = tf.keras.Input(inputSize)
        x = layers.Dense(3, activation='sigmoid')(inputs)
        outputs = layers.Dense(expOutputSize, activation='sigmoid')(x)
        models = [
            tf.keras.Model(inputs=inputs, outputs=outputs)
        ]   
    else:
        inputs = tf.keras.Input(inputSize)
        x = layers.Dense(50, activation='sigmoid')(inputs)
        x = layers.Dense(20, activation='sigmoid')(x)
        outputs = layers.Dense(expOutputSize, activation='sigmoid')(x)
        models = [
            tf.keras.Model(inputs=inputs, outputs=outputs)
        ]   
print("compiling model")

# CCE = CategoricalCrossentropy
# from_logits=True: azt jelezzük ezzel, hogy a az utolsó nem softmax() réteg, így a kimenet 'raw logit' okat tartalmaz. Ez információ alapján a CCE loss függvény először a softmax() függvényt végrehajtja a predikciókon, és csak ezután számol hibát.
# https://stackoverflow.com/a/57304538
# CCE implementációja: https://github.com/keras-team/keras/blob/985521ee7050df39f9c06f53b54e17927bd1e6ea/keras/backend/numpy_backend.py#L333
lossFunc = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metricsFunc = tf.keras.metrics.CategoricalAccuracy()
#metricsFunc = tf.keras.metrics.MeanSquaredError()
#lossFunc = tf.losses.MeanSquaredError()
for m in models:
    m.compile( 
        optimizer = tf.optimizers.Adam(),
        loss = lossFunc, 
        metrics = metricsFunc
    )
    m.summary()
#tanítás:
print("training model:")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience = 120,
    mode = 'min',
    restore_best_weights=True
)
def fit(m):
    #ha a test halmaz ures, akkor nem hasznaljuk validaciora:
    if len(featuresTest) == 0:
        history = m.fit(
            featuresTrain, 
            labelsTrain,
            batch_size=batchSize, 
            epochs=epochs_num,
            callbacks = [early_stopping]
        )
    else:
        history = m.fit(
            featuresTrain, 
            labelsTrain,
            validation_data=(featuresTest, labelsTest),
            batch_size=batchSize, 
            epochs=epochs_num,
            callbacks = [early_stopping]
        )
    return history

for i in range(0, len(models)) :
    print(str(i)+". modell tanítása:")
    print(str(fit(models[i])))
    models[i].summary()
#tesztelés:
print("testing models:")
models_and_acc = []
for m in models:
    testResult = [0.0, 0.0]

    #ha nincs test data, akkor a training data-ra vonatkozó pontosságát vetjük össze a modelleknek:
    if len(featuresTest) == 0:
        print("no validation(test) data. comparing training accuracy of models")
        testResult = m.evaluate(featuresTrain, labelsTrain, batch_size=1)
        print("also evaluating model with batchsize used for training(just for show)(loss and metrics):")
        print(m.evaluate(featuresTrain, labelsTrain, batch_size=batchSize))
    else:
        print("comparing validation accuracy of models:")
        testResult = m.evaluate(featuresTest, labelsTest, batch_size=1)
        print("also evaluating model with batchsize used for training(just for show)(loss and metrics):")
        print(m.evaluate(featuresTest, labelsTest, batch_size=batchSize))
    
    print("test results:", testResult)

    new_dict = {'model':m, 'accuracy':testResult[1]}
    print("model and accuracy:" + str(new_dict))
    models_and_acc.append(new_dict)

print("------------------------------------------------------------------------------------------")
#legjobb betanított modell kiválasztása a mért accuracy értéke alapján:
best_m_acc = max( models_and_acc, key=(lambda ma : ma['accuracy']) )
print("best model, and its accuracy: " +str(best_m_acc))
model = best_m_acc['model']
###---------------------------------------------------------------------------------------------------------------------------------------------
#ezután konvertálom át a modellbeli neurális hálót olyan formátumúra, hogy a c++ programom használhassa majd azt:
# a c++ program ilyen formában várja a neurális hálót tartalmazó file-t:
#minden rétegre(kimeneti réteget is beleszámítva): 
#   <réteg mérete>+1, placeholder , {réteg elemeinek sorozata}
#minden élsúly mátrixra: 
#   <bal réteg mérete>+1,
#   <jobb réteg mérete>+1,
#   placeholder, {bias értékek sorozata},
#   a mátrix minden oszlopára:
#       <jobb réteg mérete>+1>, placeholder, {élsúlyok sorozata}
###---------------------------------------------------------------------------------------------------------------------------------------------

print("saving most accurate model to " + saveModelPath)
model.save(saveModelPath)

print("openin "+outputFileName)
netOut = open(outputFileName,"w")
print("saving most accurate model to " +outputFileName)
#ha integrateBias 1, akkor a bias vektorok a súlymátrixok első oszlopába lesznek ágyazva, 
# és a mátrixok első sora csupa 1essel lesz kiegészítve,
# így a mátrixok szélessége és magassága 1-el megnő
integrateBias = 1


#kiírjuk a perceptron rétegek struktúráját először(csupa 1-el kitöltve a perceptronok outputjuait tároló tömböket).
#a programomban a kimeneti réteget is beleszámolom a rétegek közé így 1-el több réteg kell hogy legyen, mint élmátrix
#nem-dropout retegek megszamolasa:
lNum = 0
for lay_ in model.layers:
    if 'dropout' not in lay_.name:
        lNum += 1
#sequential model-ben az input réteg nem szerrepel a model.layers-ben, viszont általános keras.Model-ben igen, ezérrt tudni kell, melyikkel van dolgunk:
if isinstance(model, tf.keras.Sequential):
    netOut.write(  str( lNum + 1 ) + "\n"  )
    for l in range(0, len(model.layers) + 1):
        #a dropout layereket átugorjuk:
        # Check if a Layer is an Input Layer using its name
        if l < len(model.layers) and 'dropout' in model.layers[l].name:
            continue;
        #az utolsó iteráció kivételes, mert az uccsó iterációban az l. réteg nem létezik
        if l == len(model.layers):
            mx = model.layers[l-1].get_weights()[0]
            if len(mx) > 0:
                netOut.write(str( len(mx[0]) + integrateBias ) + "\n")
            #ha integrate bias 1, akkor a perceptron rétegek 1-el több elemet tartalmaznak(de a 0. elem nem igazi perceptron, csak a bias értékek foglalja a helyet)
            if integrateBias == 1:
                netOut.write("1 ")
            #a programomban a kimeneti réteget is beleszámolom a rétegek közé így 1-el több réteg kell hogy legyen, mint élmátrix,
            #és ezen réteg mérete az utolsó élmétrix másdoik dimenziójával egyezik meg(a mátrix sorainak számával):
            if len(mx) > 0 and len(mx[0]) > 0:
                for cnt in mx[0]:
                    netOut.write("0 ")
                netOut.write("\n")
        else:
            mx = model.layers[l].get_weights()[0]
            #ha integrate bias 1, akkor a perceptron rétegek 1-el több elemet tartalmaznak(de a 0. elem nem igazi perceptron, csdak a bias értékek miatt van)
            netOut.write(  str( len(mx) + integrateBias) + "\n"  )
            if integrateBias == 1:
                netOut.write("1 ")
            #a súlymátrix 1. dimenziója, azaz az oszlopok száma adja meg a hozátartozó réteg elemszámát:
            for col in mx:
                netOut.write("0 ")
            netOut.write("\n")
    netOut.write("\n")

    #a súlymátrixok  kiírása következik: 
    netOut.write(  str( lNum ) + "\n"  )
    for l in model.layers:
        #a dropout layereket átugorjuk:
        # Check if a Layer is an Input Layer using its name
        if 'dropout' in l.name:
            continue;

        biases = l.get_weights()[1]
        matrix = l.get_weights()[0]
        if len(matrix) == 0:
            continue
        #a mátrix oszlopainak száma:
        netOut.write(    str(  len(matrix) + integrateBias  ) + "\n"   )
        #ha integrateBias 1, akkor a mátrix első oszlopa a bias tömb lesz:
        if integrateBias == 1:
            #a mátrix sorainak száma:
            netOut.write(  str( len(matrix[0]) + integrateBias ) + "\n" )
            netOut.write("1 ")
            for bias in biases:
                netOut.write( str(bias) + " " )
            netOut.write("\n")
        for v in l.get_weights()[0]:
            #v az l réteghez tartozó súlymátrix oszlopain(az l rétegből KIMENŐ élek súlymátrixán) iterál
            netOut.write(  str( len(v) + integrateBias ) + "\n" )
            if integrateBias == 1:
                netOut.write("1 ")
            for weight in v:
                #w az l réteg egy súlymmátrixának v oszlopán iterál
                netOut.write( str(weight) + " ")
            netOut.write("\n")
        #ha integrateBias 0, akkor l réteg perceptronjainak bias vektorát külön írjuk ki:
        if integrateBias == 0:
            netOut.write("\n")
            netOut.write(    str(  len( l.get_weights()[1] )  )    )
            for bias in biases:
                netOut.write(str(bias) + " ")
        netOut.write("\n")
else:
    netOut.write(  str( lNum) + "\n"  )
    for l in range(1, len(model.layers) + 1):
        #a dropout layereket átugorjuk:
        # Check if a Layer is an Input Layer using its name
        if l < len(model.layers) and 'dropout' in model.layers[l].name:
            continue;
        #az utolsó iteráció kivételes, mert az uccsó iterációban az l. réteg nem létezik
        if l == len(model.layers):
            mx = model.layers[l-1].get_weights()[0]
            # map(aktivacios_func, mx*bemenet + Vbias) = kimenet
            # a bemeneti réteg mérete: len(mx)
            # a kimeneti réteg mérete: len(mx[0]) 
            if len(mx) > 0:
                netOut.write(str( len(mx[0]) + integrateBias ) + "\n")
            #ha integrate bias 1, akkor a perceptron rétegek 1-el több elemet tartalmaznak(de a 0. elem nem igazi perceptron, csak a bias értékek foglalja a helyet)
            if integrateBias == 1:
                netOut.write("1 ")
            #a programomban a kimeneti réteget is beleszámolom a rétegek közé így 1-el több réteg kell hogy legyen, mint élmátrix,
            #és ezen réteg mérete az utolsó élmátrix második dimenziójával egyezik meg(a mátrix sorainak számával):
            if len(mx) > 0 and len(mx[0]) > 0:
                for cnt in mx[0]:
                    netOut.write("0 ")
                netOut.write("\n")

        else:
            mx = model.layers[l].get_weights()[0]
            #ha integrate bias 1, akkor a perceptron rétegek 1-el több elemet tartalmaznak(de a 0. elem nem igazi perceptron, csak a bias értékek miatt van)
            netOut.write(  str( len(mx) + integrateBias) + "\n"  )
            if integrateBias == 1:
                netOut.write("1 ")
            #a súlymátrix 1. dimenziója, azaz az oszlopok száma adja meg a hozzá tartozó réteg elemszámát:
            for col in mx:
                netOut.write("0 ")
            netOut.write("\n")
    netOut.write("\n")

    #a súlymátrixok  kiírása következik: 
    netOut.write(  str( lNum-1 ) + "\n"  )
    for l in model.layers[1:]:
        #a dropout layereket átugorjuk:
        # Check if a Layer is an Input Layer using its name
        if 'dropout' in l.name:
            continue;

        biases = l.get_weights()[1]
        matrix = l.get_weights()[0]
        if len(matrix) == 0:
            continue
        #a mátrix oszlopainak száma:
        netOut.write(    str(  len(matrix) + integrateBias  ) + "\n"   )
        #ha integrateBias 1, akkor a mátrix első oszlopa a bias tömb lesz:
        if integrateBias == 1:
            #a mátrix sorainak száma:
            netOut.write(  str( len(matrix[0]) + integrateBias ) + "\n" )
            netOut.write("1 ")
            for bias in biases:
                netOut.write( str(bias) + " " )
            netOut.write("\n")
        for v in l.get_weights()[0]:
            #v az l réteghez tartozó súlymátrix oszlopain iterál
            netOut.write(  str( len(v) + integrateBias ) + "\n" )
            if integrateBias == 1:
                netOut.write("1 ")
            for weight in v:
                #w az l réteg egy súlymmátrixának v oszlopának elemein iterál, tehát a mátrix egy oszlopa a file egy sorának fog megfelelni
                netOut.write( str(weight) + " ")
            netOut.write("\n")
        #ha integrateBias 0, akkor l réteg perceptronjainak bias vektorát külön írjuk ki:
        if integrateBias == 0:
            netOut.write("\n")
            netOut.write(    str(  len( l.get_weights()[1] )  )    )
            for bias in biases:
                netOut.write(str(bias) + " ")
        netOut.write("\n")
netOut.close()
###---------------------------------------------------------------------------------------------------------------------------------------------

sys.stdout.close()
sys.stderr.close()