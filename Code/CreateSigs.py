import pandas as pd
import numpy as np
import aifc as aifc
import matplotlib.pyplot as plt
import os as os
from pylab import *
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn import preprocessing

baseDir =  "/home/ddmm/Desktop/Kaggle/Data"
trainingFile = "train.csv"

trainTruth = pd.read_csv(os.path.join(baseDir, trainingFile))

nData=30000

#these constants gave 91.8% AUC
frequencySlice = range(10, 50)
timeSlice      = range(4, 21)

signalCount    = 0
# primaryCount = 0
# lowCount     = 0
# downCount    = 0

signalModel             = zeros((len(frequencySlice),len(timeSlice)))
signalModelDeMeaned     = zeros((len(frequencySlice),len(timeSlice)))
# primaryModel          = zeros((len(frequencySlice),len(timeSlice)))
# primaryModelDeMeaned  = zeros((len(frequencySlice),len(timeSlice)))
# lowTypeModel          = zeros((len(frequencySlice),len(timeSlice)))
# lowTypeModelDeMeaned  = zeros((len(frequencySlice),len(timeSlice)))
# downTypeModel         = zeros((len(frequencySlice),len(timeSlice)))
# downTypeModelDeMeaned = zeros((len(frequencySlice),len(timeSlice)))

# badSignatureList = [673, 795, 1127, 1225, 2166, 2179, 2187, 2191, 2232, 2689, 2722, 3032, 3548,
						# 3675, 3775, 3783, 3810, 3973]

#confuserBackList = [3, 4, 5, 8, 85]


#primaryTypeList = [6, 7, 9, 180, 379, 880, 1112]
#lowTypeList     = [12, 133, 227, 229, 230, 509, 701, 1142, 1143, 1148, 1150, 1174, 1330, 1442]
#downTypeList    = [77, 418, 556]
#createSigList   = [55, 352, 509, 727, 893, 997]
#createSigList   = [1000, 1051, 1101, 1174, 2187]
#createSigList   = [32, 79, 180, 651, 730, 3147, 3560, 5225, 5264, 6323, 6335]
createSigList    = [6323]
#DRL these are  

# badSignatureList = ["train"+str(a)+".aiff" for a in badSignatureList]
# confuserBackList = ["train"+str(a)+".aiff" for a in badSignatureList]
# primaryTypeList  = ["train"+str(a)+".aiff" for a in primaryTypeList]
# lowTypeList      = ["train"+str(a)+".aiff" for a in lowTypeList]
# downTypeList     = ["train"+str(a)+".aiff" for a in downTypeList]
createSigList      = ["train"+str(a)+".aiff" for a in createSigList]

for x in range(0, nData):
	
	if trainTruth["clip_name"][x] in createSigList:
		#set the filename based on the trainTruth clip_name
		currFile = os.path.join(baseDir, "train", trainTruth["clip_name"][x])
		
		#read in the current file
		reader  = aifc.open(currFile, 'r')
		nframes = reader.getnframes()
		
		strsig   = reader.readframes(nframes)
		currData = np.fromstring(strsig, np.short).byteswap()
		
		#Calculate the spectrogram of the audio
		currSpec = specgram(currData)
		
		#Pull out the magnitude of the time/frequency slice where the signal resides
		# as well as a background slice in the frequency slice before and after the signal
		spectraSlice    = currSpec[0][frequencySlice,:][:,timeSlice]
		backgroundSlice = currSpec[0][frequencySlice,:][:,(0,1,2,3,26,27,28,29)]
		
		#Calculate the mean per frequency bin across time
		# and thread it through a second dimension so we can do vector arithmetic
		#backgroundMean = np.mean(backgroundSlice, axis=1)
		backgroundMean = np.mean(backgroundSlice, axis=1)
		backgroundMean = backgroundMean.reshape(len(backgroundMean), 1)
		
		#Subtract the background from the spectra
		deMeanedSpectra = spectraSlice - backgroundMean
		
		#clip the spectra to remove negative numbers (actually clip at 1)  and then take the log
		clippedSpectra = deMeanedSpectra.clip(1)
		logClippedSpectra = log(clippedSpectra)
		
		plt.title(trainTruth["clip_name"][x]+"Sig")
		plt.imshow(logClippedSpectra.clip(7))
		plt.gca().invert_yaxis()
		draw()
		savefig(os.path.join(baseDir, "Images",  trainTruth["clip_name"][x] + "logClipped7.png"))
		np.savetxt(os.path.join(baseDir, "signatures", trainTruth["clip_name"][x]+".csv"), preprocessing.scale(logClippedSpectra.clip(11)), delimiter=",")

	
	# if (trainTruth["label"][x] == 1 and trainTruth["clip_name"][x] not in badSignatureList):
		# #aggregate the truth data to create a signal model
		# signalModel += spectraSlice
		# signalModelDeMeaned += clippedSpectra
		# signalCount = signalCount + 1
		# print "Signal count: %d"%signalCount
		
	# if trainTruth["clip_name"][x] in primaryTypeList:
		# #aggregate the truth data to create the primary type signal model
		# primaryModel += spectraSlice
		# primaryModelDeMeaned += clippedSpectra
		# primaryCount = primaryCount + 1
		# print "primaryCount count: %d"%primaryCount
	
	# if trainTruth["clip_name"][x] in lowTypeList:
		# #aggregate the truth data to create the primary type signal model
		# lowTypeModel += spectraSlice
		# lowTypeModelDeMeaned += clippedSpectra
		# lowCount = lowCount + 1
		# print "lowCount count: %d"%lowCount
		
	# if trainTruth["clip_name"][x] in downTypeList:
		# #aggregate the truth data to create the primary type signal model
		# downTypeModel += spectraSlice
		# downTypeModelDeMeaned += clippedSpectra
		# downCount = downCount + 1
		# print "downCount count: %d"%downCount
		
# signalModel /= signalCount
# signalModelDeMeaned /= signalCount

# primaryModel                    /= primaryCount
# primaryModelDeMeaned     /= primaryCount
# lowTypeModel                   /= lowCount
# lowTypeModelDeMeaned    /= lowCount
# downTypeModel                /= downCount
# downTypeModelDeMeaned /= downCount

# #Write the ensembled signal
# ax7 = plt.subplot2grid((3,3), (0,0), colspan=3, rowspan=3)
# plt.title("Ensembled Signal")
# plt.imshow(log(signalModel))
# plt.gca().invert_yaxis()
# draw()
# savefig(os.path.join(baseDir, "images",  trainTruth["clip_name"][x] + "Ensembled.png"))

# #Write the ensembled demeaned signal
# plt.title("Ensembled Demeaned Signal")
# plt.imshow(log(signalModelDeMeaned))
# draw()
# savefig(os.path.join(baseDir, "images",  trainTruth["clip_name"][x] + "EnsembledDemeaned.png"))

# plt.title("Ensembled primary Signal")
# plt.imshow(log(primaryModel))
# draw()
# savefig(os.path.join(baseDir, "images",  trainTruth["clip_name"][x] + "primaryModel.png"))

# plt.title("Ensembled Demeaned primary Signal")
# plt.imshow(log(primaryModelDeMeaned))
# draw()
# savefig(os.path.join(baseDir, "images",  trainTruth["clip_name"][x] + "primaryModelDeMeaned.png"))

# plt.title("Ensembled low Signal")
# plt.imshow(log(lowTypeModel))
# draw()
# savefig(os.path.join(baseDir, "images",  trainTruth["clip_name"][x] + "lowTypeModel.png"))

# plt.title("Ensembled Demeaned low Signal")
# plt.imshow(log(lowTypeModelDeMeaned))
# draw()
# savefig(os.path.join(baseDir, "images",  trainTruth["clip_name"][x] + "lowTypeModelDeMeaned.png"))

# plt.title("Ensembled down Signal")
# plt.imshow(log(downTypeModel))
# draw()
# savefig(os.path.join(baseDir, "images",  trainTruth["clip_name"][x] + "downTypeModel.png"))

# plt.title("Ensembled Demeaned down Signal")
# plt.imshow(log(downTypeModelDeMeaned))
# draw()
# savefig(os.path.join(baseDir, "images",  trainTruth["clip_name"][x] + "downTypeModelDeMeaned.png"))

# np.savetxt(os.path.join(baseDir, "signatures", "EnsembledSignature"+str(nData)+".csv"), signalModel, delimiter=",")
# np.savetxt(os.path.join(baseDir, "signatures", "EnsembledSignatureDeMeaned"+str(nData)+".csv"), signalModelDeMeaned, delimiter=",")

# np.savetxt(os.path.join(baseDir, "signatures", "primaryModel"+str(nData)+".csv"), signalModel, delimiter=",")
# np.savetxt(os.path.join(baseDir, "signatures", "primaryModelDeMeaned"+str(nData)+".csv"), signalModelDeMeaned, delimiter=",")

# np.savetxt(os.path.join(baseDir, "signatures", "lowTypeModel"+str(nData)+".csv"), signalModel, delimiter=",")
# np.savetxt(os.path.join(baseDir, "signatures", "lowTypeModelDeMeaned"+str(nData)+".csv"), signalModelDeMeaned, delimiter=",")

# np.savetxt(os.path.join(baseDir, "signatures", "downTypeModel"+str(nData)+".csv"), signalModel, delimiter=",")
# np.savetxt(os.path.join(baseDir, "signatures", "downTypeModelDeMeaned"+str(nData)+".csv"), signalModelDeMeaned, delimiter=",")



