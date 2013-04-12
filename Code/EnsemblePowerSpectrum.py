import pandas as pd
import numpy as np
import aifc as aifc
import matplotlib.pyplot as plt
import os as os
import msvcrt as m
from pylab import *
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn import preprocessing
import WhaleUtils

reload(WhaleUtils)

baseDir =  "C:\\Users\\DDMM\\Desktop\\Kaggle\\Whale\\data"
trainingFile = "train.csv"

trainTruth = pd.read_csv(os.path.join(baseDir, trainingFile))

nData=30000

ensembledBack = np.zeros(2001)

CleanBackList = [42, 49, 83, 91, 125, 150, 170, 178, 223, 292, 338, 372, 377, 407, 410
							, 412, 414, 415, 421, 422, 431, 435, 440, 441, 440, 445, 447, 491, 532
							, 538, 560, 561, 567, 578, 579, 593, 595, 693, 698, 749, 753, 759, 760
							, 767, 776, 803, 809, 819, 822, 826, 831, 832, 833, 835, 836, 837, 838
							, 843, 885, 897]

CleanBackList = ["train"+str(a)+".aiff" for a in CleanBackList]				

for x in range(nData):
	
	if trainTruth["clip_name"][x] in CleanBackList:
		
		#set the filename based on the trainTruth clip_name
		currFile = os.path.join(baseDir, "train", trainTruth["clip_name"][x])
		
		#read in the current file
		reader = aifc.open(currFile, 'r')
		nframes = reader.getnframes()
		
		strsig = reader.readframes(nframes)
		currData = np.fromstring(strsig, np.short).byteswap()
		
		backFFT = np.fft.rfft(currData)
		magBackFFT = abs(backFFT)
		
		ensembledBack += magBackFFT
		
ensembledBack = ensembledBack / len(CleanBackList)
smoothedBack = WhaleUtils.smooth(ensembledBack, 100)
smoothedBack = smoothedBack[49:2050]

plt.plot(ensembledBack)
plt.plot(smoothedBack)
draw()
show()

np.savetxt(os.path.join(baseDir, "powerSpectra", "EnsembledSpectra_Iter1.csv"), smoothedBack, delimiter=",")