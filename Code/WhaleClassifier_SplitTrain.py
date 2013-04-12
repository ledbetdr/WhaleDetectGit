import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import os as os
#import msvcrt as m
from pylab import *
from scipy import signal
from scipy import polyval, polyfit
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn import preprocessing
import WhaleUtils
import WhaleUtils_Segment

reload(WhaleUtils)
reload(WhaleUtils_Segment)

calcTest = True
usePriors = True

baseDir    = "/home/ddmm/Desktop/Kaggle/Data"
featureCSV = os.path.join(baseDir, "baseFeatures.csv")
trainFiles = pd.read_csv(os.path.join(baseDir, "train.csv"))
testFiles  = pd.read_csv(os.path.join(baseDir, "test.csv"))
trainCSV   = os.path.join(baseDir, "CSVs/trainFeatures.csv")
testCSV	   = os.path.join(baseDir, "CSVs/testFeatures.csv")
trainDir   = os.path.join(baseDir, "train")
testDir    = os.path.join(baseDir, "test")
sigDir     = os.path.join(baseDir, "signatures")

#The range of the subarray to slice out of the spectrogram
#frequencySlice = range(10, 50)
frequencySlice = range(10, 50)
#timeSlice      = range(4, 21)
timeSlice      = range(2, 23)

#The lower bound we'll clip to remove noise elements
logClipping = 1

#Amount of data to read in (full dataset is 30k)
nData = 30000
nTest = 54503

#split the data into segments with similar priors
trainingSplits = np.array([0, 3600, 13000, 18400, 27400, 30000])
testSplits = np.array(trainingSplits.astype(float) / 30000 * 54503).astype(int)

#Pick your favourite classifier
#clf = RandomForestClassifier(n_estimators=100, n_jobs=1, compute_importances = True)
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.02, max_depth=11, random_state=0, subsample=.5)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.05, max_depth=6, random_state=0, subsample=.5)
clf = GradientBoostingClassifier(n_estimators=5000, learning_rate=.001, max_depth=10, random_state=0, subsample=.5)
#clf = GradientBoostingClassifier(n_estimators=10000, learning_rate=.002, max_depth=10, random_state=0, subsample=.5)
#clf = ExtraTreesClassifier(n_estimators=100)
#clf = svm.SVC(kernel='rbf', C=1.0, shrinking=True, probability=True)

#let's create a matrix that we can use for training
# we'll want it to be len(frequencySlice)xlen(timeSlice) long
# we'll append each new array to organize our training features
#featureList = WhaleUtils.defFeatureList(len(frequencySlice), len(timeSlice))


#generate the training csv file
trainFeatures = WhaleUtils.generateFeatureCSV(trainFiles, trainDir, baseDir, nData, frequencySlice, 
						timeSlice, logClipping, trainCSV)
#trainFeatures = read_csv(featureCSV, index_col=0, header=0)
#trainFeatures = WhaleUtils_Segment.generateFeatureCSV(trainFiles, trainDir, baseDir, nData, frequencySlice, 
						#timeSlice, logClipping, trainCSV)

if calcTest:
	testFeatures = WhaleUtils.generateFeatureCSV(testFiles, testDir, baseDir, nTest, frequencySlice,
							timeSlice, logClipping, testCSV)
    #testFeatures = WhaleUtils_Segment.generateFeatureCSV(testFiles, testDir, baseDir, nTest, frequencySlice,
							#timeSlice, logClipping, testCSV)
    

#set the random seed
random_state = np.random.RandomState(1)

if usePriors:
	trainPriors, testPriors = WhaleUtils.populatePriors(trainFiles["label"], smoothWindow=50)

	trainFeatures['priors'] = trainPriors

	if calcTest:
		testFeatures['priors'] = testPriors

#If you were using a classifier that needed unit mean/var you can use
# the preprocessing.scale np function
#dataArray_scaled = preprocessing.scale(dataArray)

#trainingSplits = np.array([0, 3500, 7500, 13000, 18400, 27400, 30000])
#trainingSplits = np.array([0, 3600, 13000, 18400, 27400, 30000])
#testSplits = np.array(trainingSplits.astype(float) / 30000 * 54503).astype(int)

predictions = np.array([])
y_appended = np.array([])

for index in np.arange(0, len(trainingSplits)-1):
	if not calcTest:
		#Shuffle our data based on the random seed and set it into X [the feature vector] and y [the class]
		#X, y = shuffle(dataArray, trainFiles["label"][0:nData], random_state=random_state)
		#X, y = shuffle(dataArray_scaled, trainFiles["label"][0:nData], random_state=random_state)
		X, y = shuffle(trainFeatures[trainingSplits[index]:trainingSplits[index+1]], trainFiles["label"][trainingSplits[index]:trainingSplits[index+1]], random_state=0)

		#Split the data in half to train/test
		half = int((trainingSplits[index + 1] - trainingSplits[index]) / 2)

		#Set the first half of the data to train and the second half to test
		X_train, X_test = X[:half], X[half:]
		y_train, y_test = y[:half], y[half:]
		
		y_appended = np.append(y_appended, y_test)
	else:
		X, y = shuffle(trainFeatures[trainingSplits[index]:trainingSplits[index+1]], trainFiles["label"][trainingSplits[index]:trainingSplits[index+1]], random_state=0)

		X_train = X
		y_train = y

		X_test = testFeatures[testSplits[index]:testSplits[index+1]]

	print("Training...")

	#Fit the classifier
	clf.fit(X_train, y_train)

	print("Testing...")

	#Predict probabilities on test data
	tmpPredictions = clf.predict_proba(X_test)
	predictions = np.append(predictions, tmpPredictions[:,1])

if not calcTest:
	#Create a ROC curve and calculate the area under the curve (AUC)
	# AUC is the metric by which this competition is judged!
	fpr, tpr, thresholds = roc_curve(y_appended, predictions)
	#fpr, tpr, thresholds = roc_curve(y_train, predictions[:,1])
	roc_auc = auc(fpr, tpr)
	print "Area under the ROC curve : %f" % roc_auc

	# Plot ROC curve
	plt.clf()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Right Whale ROC curve')
	plt.legend(loc="lower right")
	plt.show()

#show the top 20 most important features
importance = sorted(zip(trainFeatures.columns, clf.feature_importances_), key=lambda tup: tup[1], reverse=True)

#run classifier on full dataframe to pull out false targets and missed detections
if False:
    fullPredictions = clf.predict_proba(trainFeatures)
    #histogram of the target and FA probabilities
    tgtIndeces = trainFiles["label"][0:len(trainFeatures)]==1
    faIndeces = trainFiles["label"][0:len(trainFeatures)]==0

    n, bins, patches = plt.hist(fullPredictions[faIndeces, 1], 150, facecolor='red', alpha=.6)
    n, bins, patches = plt.hist(fullPredictions[tgtIndeces, 1], 150, facecolor='green', alpha=.6)

    #spit out csv of full feature field for data exploration
    trainFeatures["probabilities"] = fullPredictions[:,1]
    trainFeatures["label"] = np.asarray(trainFiles["label"][0:len(trainFeatures)])
    trainFeatures.to_csv("WhaleData_WithSegmentation.csv")
	
if calcTest:
	#testFeatures["probabilities"] = testPredictions[:,1]
	submission = pd.DataFrame(predictions, index=testFiles["clip_name"][0:nTest])
	submission.to_csv("TestSubmission_Iter7_depth11_prior50_lr04)500m.csv")




