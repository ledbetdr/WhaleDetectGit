from scipy import signal
from scipy import polyval, polyfit
from sklearn import preprocessing
import numpy as np
from numpy import unravel_index
import os as os
from pylab import *
import pandas as pd
import aifc as aifc
import scipy as sp
from scipy import ndimage
from PIL import Image
import pymeanshift as pms
from scipy.sparse import *
from scipy import stats

def loadSig(baseDir, sigFile):
	#Read in and normalize the signature
	sig = preprocessing.normalize(np.genfromtxt(os.path.join(baseDir, "signatures", sigFile), delimiter=","))
	
	#Add negative power to the background of the sig
	#sig[(sig==0)] = -1 * np.sum(sig) / np.sum(sig!=0)
	
	#Set total power of signal to 1
	sig = sig /  np.sum(sig)
	
	#Flatten the signature  so I can perform a vector multiplication
	sigFlat = np.array(list(sig.flatten()))
	
	#Calculate the translation constant of the log likelihood
	sigTranslationConst = np.sum(sigFlat * sigFlat)
	#sigTranslationConst = 0
	
	#return signature and the translation constant
	return sig, sigTranslationConst
	
def createH0SigHorizontalLine(length, width):
	h0Sig = np.zeros((width + 2, length))
	h0Sig[1:-1,:] =  1
	h0Sig[0,:]     = -1
	h0Sig[-1,:]    = -1
	h0Sig = preprocessing.scale(h0Sig)
	return h0Sig

def createH0SigVerticalLine(length, width):
	h0Sig = np.zeros((length, width + 2))
	h0Sig[:,1:-1] =  1
	h0Sig[:,0]      = -1
	h0Sig[:,-1]     = -1
	h0Sig = preprocessing.scale(h0Sig)
	return h0Sig
	
def createH0SigGaussianBlob(size):
	x, y = mgrid[-size:size+1, -size:size+1]
	h0Sig = exp(-(x**2/float(size)+y**2/float(size)))
	return h0Sig / h0Sig.sum()

	
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def defFeatureList(freqLength, timeLength):
	featureList = []

	featureList.append("maxMatchedFilter")
	featureList.append("matchedFilter1")
	featureList.append("matchedFilter2")
	featureList.append("matchedFilter3")
	featureList.append("matchedFilter4")
	featureList.append("matchedFilter5")
	featureList.append("matchedFilter6")
	featureList.append("matchedFilter7")
	featureList.append("matchedFilter8")
	featureList.append("matchedFilter9")
	featureList.append("matchedFilter10")
	featureList.append("matchedFilter11")
	featureList.append("matchedFilter12")
	featureList.append("matchedFilter13")
	featureList.append("matchedFilter14")
	featureList.append("matchedFilter15")
	featureList.append("matchedFilter16")
	featureList.append("matchedFilter17")
	
	featureList.append("matchedFilter1x")
	featureList.append("matchedFilter2x")
	featureList.append("matchedFilter3x")
	featureList.append("matchedFilter4x")
	featureList.append("matchedFilter5x")
	featureList.append("matchedFilter6x")
	featureList.append("matchedFilter7x")
	featureList.append("matchedFilter8x")
	featureList.append("matchedFilter9x")
	featureList.append("matchedFilter10x")
	featureList.append("matchedFilter11x")
	featureList.append("matchedFilter12x")
	featureList.append("matchedFilter13x")
	featureList.append("matchedFilter14x")
	featureList.append("matchedFilter15x")
	featureList.append("matchedFilter16x")
	featureList.append("matchedFilter17x")
	
	featureList.append("matchedFilter1y")
	featureList.append("matchedFilter2y")
	featureList.append("matchedFilter3y")
	featureList.append("matchedFilter4y")
	featureList.append("matchedFilter5y")
	featureList.append("matchedFilter6y")
	featureList.append("matchedFilter7y")
	featureList.append("matchedFilter8y")
	featureList.append("matchedFilter9y")
	featureList.append("matchedFilter10y")
	featureList.append("matchedFilter11y")
	featureList.append("matchedFilter12y")
	featureList.append("matchedFilter13y")
	featureList.append("matchedFilter14y")
	featureList.append("matchedFilter15y")
	featureList.append("matchedFilter16y")
	featureList.append("matchedFilter17y")
	
	featureList.append("avgMatchedFilterX")
	featureList.append("avgMatchedFilterY")
	
	featureList.append("maxMatchedFilterCorr")
	featureList.append("matchedFilter1Correlation")
	featureList.append("matchedFilter2Correlation")
	featureList.append("matchedFilter3Correlation")
	featureList.append("matchedFilter4Correlation")
	featureList.append("matchedFilter5Correlation")
	featureList.append("matchedFilter6Correlation")
	featureList.append("matchedFilter7Correlation")
	featureList.append("matchedFilter8Correlation")
	featureList.append("matchedFilter9Correlation")
	featureList.append("matchedFilter10Correlation")
	featureList.append("matchedFilter11Correlation")
	featureList.append("matchedFilter12Correlation")
	featureList.append("matchedFilter13Correlation")
	featureList.append("matchedFilter14Correlation")
	featureList.append("matchedFilter15Correlation")
	featureList.append("matchedFilter16Correlation")
	featureList.append("matchedFilter17Correlation")
	
	featureList.append("logClippedmaxMatchedFilter")
	featureList.append("logClippedmatchedFilter1")
	featureList.append("logClippedmatchedFilter2")
	featureList.append("logClippedmatchedFilter3")
	featureList.append("logClippedmatchedFilter4")
	featureList.append("logClippedmatchedFilter5")
	featureList.append("logClippedmatchedFilter6")
	featureList.append("logClippedmatchedFilter7")
	featureList.append("logClippedmatchedFilter8")
	featureList.append("logClippedmatchedFilter9")
	featureList.append("logClippedmatchedFilter10")
	featureList.append("logClippedmatchedFilter11")
	featureList.append("logClippedmatchedFilter12")
	featureList.append("logClippedmatchedFilter13")
	featureList.append("logClippedmatchedFilter14")
	featureList.append("logClippedmatchedFilter15")
	featureList.append("logClippedmatchedFilter16")
	featureList.append("logClippedmatchedFilter17")

	featureList.append("deMeanedmaxMatchedFilter")
	featureList.append("deMeanedmatchedFilter1")
	featureList.append("deMeanedmatchedFilter2")
	featureList.append("deMeanedmatchedFilter3")
	featureList.append("deMeanedmatchedFilter4")
	featureList.append("deMeanedmatchedFilter5")
	featureList.append("deMeanedmatchedFilter6")
	featureList.append("deMeanedmatchedFilter7")
	featureList.append("deMeanedmatchedFilter8")
	featureList.append("deMeanedmatchedFilter9")
	featureList.append("deMeanedmatchedFilter10")
	featureList.append("deMeanedmatchedFilter11")
	featureList.append("deMeanedmatchedFilter12")
	featureList.append("deMeanedmatchedFilter13")
	featureList.append("deMeanedmatchedFilter14")
	featureList.append("deMeanedmatchedFilter15")
	featureList.append("deMeanedmatchedFilter16")
	featureList.append("deMeanedmatchedFilter17")
	
	featureList.append("maxWhitenedMatchedFilter")
	featureList.append("whitenedMatchedFilter1")
	featureList.append("whitenedMatchedFilter2")
	featureList.append("whitenedMatchedFilter3")
	featureList.append("whitenedMatchedFilter4")
	featureList.append("whitenedMatchedFilter5")
	featureList.append("whitenedMatchedFilter6")
	featureList.append("whitenedMatchedFilter7")
	featureList.append("whitenedMatchedFilter8")
	featureList.append("whitenedMatchedFilter9")
	featureList.append("whitenedMatchedFilter10")
	featureList.append("whitenedMatchedFilter11")
	featureList.append("whitenedMatchedFilter12")
	featureList.append("whitenedMatchedFilter13")
	featureList.append("whitenedMatchedFilter14")
	featureList.append("whitenedMatchedFilter15")
	featureList.append("whitenedMatchedFilter16")
	featureList.append("whitenedMatchedFilter17")
	
	featureList.append("H0matchedFilter1")
	featureList.append("H0matchedFilter2")
	#featureList.append("H0matchedFilter3")
	#featureList.append("H0matchedFilter4")
	featureList.append("H0matchedFilter5")
	featureList.append("H0matchedFilter6")

	featureList.append("sliceMean")
	featureList.append("sliceVar")
	featureList.append("backMean")
	featureList.append("backVar")
	featureList.append("outMean")

	featureList.append("sliceBackRatio")
	featureList.append("sliceOutRatio")
	featureList.append("clippedMean")
	#featureList.append("clippedVar")
	featureList.append("logClippedMean5")
	featureList.append("logClippedVar5")
	featureList.append("logClippedMean10")
	featureList.append("logClippedVar10")
	featureList.append("logClippedMean13")
	featureList.append("logClippedVar13")
	
	#featureList.append("v")
	#featureList.append("minI")
	#featureList.append("maxI")
	#featureList.append("minT")
	#featureList.append("maxT")
	#featureList.append("minF")
	#featureList.append("maxF")
	#featureList.append("maxTminusminT")
	#featureList.append("maxFminusminF")
	
	for x in range(freqLength):
		featureList.append("Col"+str(x)+"Mean")
	
	for x in range(freqLength):
		featureList.append("Col"+str(x)+"Var")
		
	for x in range(freqLength):
		featureList.append("Col"+str(x)+"Skew")
		
	for x in range(timeLength):
		featureList.append("Row"+str(x)+"Mean")

	for x in range(timeLength):
		featureList.append("Row"+str(x)+"Var")
		
	for x in range(timeLength):
		featureList.append("Row"+str(x)+"Skew")
		
	return featureList

def generateFeatureCSV(clipInfoCSV, dataDir, baseDir, nData, frequencySlice, timeSlice, logClipping, trainCSV):
	#create an empty array we'll store all the data in
	dataArray = []

	#Read in all the signatures we'll use for matched filtering/whitening
	sig1, sig1TranslationConst = loadSig(baseDir, "Sig1_20x13_From55.csv")
	sig2, sig2TranslationConst = loadSig(baseDir, "Sig2_13x12_From352.csv")
	sig3, sig3TranslationConst = loadSig(baseDir, "Sig3_15x11_From509.csv")
	sig4, sig4TranslationConst = loadSig(baseDir, "Sig4_10x10_from727.csv")
	sig5, sig5TranslationConst = loadSig(baseDir, "Sig5_10x9_From997.csv")
	sig6, sig6TranslationConst = loadSig(baseDir, "Sig6_10x7_From1000.csv")
	sig7, sig7TranslationConst = loadSig(baseDir, "Sig7_19x13_From1051.csv")
	sig8, sig8TranslationConst = loadSig(baseDir, "Sig8_21x5_From1101.csv")
	sig9, sig9TranslationConst = loadSig(baseDir, "Sig9_14x12_From1174.csv")

	sig10, sig10TranslationConst = loadSig(baseDir, "Sig10_18x7_from32.csv")
	sig11, sig11TranslationConst = loadSig(baseDir, "Sig11_20x5_From79.csv")
	sig12, sig12TranslationConst = loadSig(baseDir, "Sig12_21x8_From180.csv")
	sig13, sig13TranslationConst = loadSig(baseDir, "Sig13_21x8_from651.csv")
	sig14, sig14TranslationConst = loadSig(baseDir, "Sig14_16x4_From730.csv")
	sig15, sig15TranslationConst = loadSig(baseDir, "Sig15_8x13_From3147.csv")
	sig16, sig16TranslationConst = loadSig(baseDir, "Sig16_12x10_From5225.csv")
	sig17, sig17TranslationConst = loadSig(baseDir, "Sig17_21x15_From6323.csv")

	h01 = createH0SigHorizontalLine(len(timeSlice), 1)
	h02 = createH0SigHorizontalLine(len(timeSlice), 3)
	#h03 = createH0SigVerticalLine(len(frequencySlice), 1)
	#h04 = createH0SigVerticalLine(len(frequencySlice), 3)
	h05 = createH0SigGaussianBlob(3)
	h06 = createH0SigGaussianBlob(5)
		
	ensembledPowerSpectra = np.genfromtxt(os.path.join(baseDir, "powerSpectra", "EnsembledSpectra_Iter1.csv"), delimiter=",")
	
	#read in data
	for x in range(nData):
	
		#set the filename based on the trainTruth clip_name
		currFile = os.path.join(dataDir, clipInfoCSV["clip_name"][x])
	
		#read in the current file
		reader = aifc.open(currFile, 'r')
		nframes = reader.getnframes()
	
		strsig = reader.readframes(nframes)
		currData = np.fromstring(strsig, np.short).byteswap()
	
		#whiten the data
		testFFT = np.fft.rfft(currData)
		testFFT = testFFT / ensembledPowerSpectra
		whitenedData = np.fft.irfft(testFFT)
	
		#Calculate the spectrogram of the audio
		currSpec = specgram(currData)
		whitenedSpec = specgram(whitenedData)
	
		#Pull out the magnitude of the time/frequency slice where the signal resides
		# as well as a background slice in the frequency slice before and after the signal
		spectraSlice = currSpec[0][frequencySlice,:][:,timeSlice]
		backgroundSlice = currSpec[0][frequencySlice,:][:,(0,1,2,3,26,27,28,29)]
		whitenedSpectraSlice = whitenedSpec[0][frequencySlice,:][:,timeSlice]
	
		#Calculate the mean per frequency bin across time
		# and thread it through a second dimension so we can do vector arithmetic
		#backgroundMean = np.mean(backgroundSlice, axis=1)
		backgroundMean = np.mean(backgroundSlice, axis=1)
		backgroundMean = backgroundMean.reshape(len(backgroundMean), 1)
	
		# backgroundVar = np.var(backgroundSlice, axis=1)
		# backgroundVar = backgroundVar.reshape(len(backgroundMean), 1)
	
		#Subtract the background from the spectra
		deMeanedSpectra = (spectraSlice - backgroundMean)
	
		#clip the spectra to remove negative numbers (actually clip at 1)  and then take the log
		clippedSpectra = deMeanedSpectra.clip(1)
		logClippedSpectra = log(clippedSpectra)
	
		#Calculate the matched filteres by convolving over the spectra slice with our various signatures
		# We'll then 'pool' by taking the max convolution over the region
		# Each of these matched filters will become input as well as a 'maxFilter'
		#normalizedSpectra = preprocessing.normalize(log(spectraSlice))
		normalizedSpectra = preprocessing.normalize(log(spectraSlice))
		scaledDeMeanedSpectra = preprocessing.scale(deMeanedSpectra)
		scaledClippedSpectra = preprocessing.scale(logClippedSpectra.clip(10))
		normWhitenedSpectraSlice = preprocessing.normalize(whitenedSpectraSlice)
		
		tmpMatchedFilter1 = sp.signal.convolve2d(normalizedSpectra, sig1, mode='same')
		tmpMatchedFilter2 = sp.signal.convolve2d(normalizedSpectra, sig2, mode='same')
		tmpMatchedFilter3 = sp.signal.convolve2d(normalizedSpectra, sig3, mode='same')
		tmpMatchedFilter4 = sp.signal.convolve2d(normalizedSpectra, sig4, mode='same')
		tmpMatchedFilter5 = sp.signal.convolve2d(normalizedSpectra, sig5, mode='same')
		tmpMatchedFilter6 = sp.signal.convolve2d(normalizedSpectra, sig6, mode='same')
		tmpMatchedFilter7 = sp.signal.convolve2d(normalizedSpectra, sig7, mode='same')
		tmpMatchedFilter8 = sp.signal.convolve2d(normalizedSpectra, sig8, mode='same')
		tmpMatchedFilter9 = sp.signal.convolve2d(normalizedSpectra, sig9, mode='same')
		tmpMatchedFilter10 = sp.signal.convolve2d(normalizedSpectra, sig10, mode='same')
		tmpMatchedFilter11 = sp.signal.convolve2d(normalizedSpectra, sig11, mode='same')
		tmpMatchedFilter12 = sp.signal.convolve2d(normalizedSpectra, sig12, mode='same')
		tmpMatchedFilter13 = sp.signal.convolve2d(normalizedSpectra, sig13, mode='same')
		tmpMatchedFilter14 = sp.signal.convolve2d(normalizedSpectra, sig14, mode='same')
		tmpMatchedFilter15 = sp.signal.convolve2d(normalizedSpectra, sig15, mode='same')
		tmpMatchedFilter16 = sp.signal.convolve2d(normalizedSpectra, sig16, mode='same')
		tmpMatchedFilter17 = sp.signal.convolve2d(normalizedSpectra, sig17, mode='same')
	
		matchedFilter1 = np.amax(tmpMatchedFilter1) - sig1TranslationConst
		matchedFilter2 = np.amax(tmpMatchedFilter2) - sig2TranslationConst
		matchedFilter3 = np.amax(tmpMatchedFilter3) - sig3TranslationConst
		matchedFilter4 = np.amax(tmpMatchedFilter4) - sig4TranslationConst
		matchedFilter5 = np.amax(tmpMatchedFilter5) - sig5TranslationConst
		matchedFilter6 = np.amax(tmpMatchedFilter6) - sig6TranslationConst
		matchedFilter7 = np.amax(tmpMatchedFilter7) - sig7TranslationConst
		matchedFilter8 = np.amax(tmpMatchedFilter8) - sig8TranslationConst
		matchedFilter9 = np.amax(tmpMatchedFilter9) - sig9TranslationConst
		matchedFilter10 = np.amax(tmpMatchedFilter10) - sig10TranslationConst
		matchedFilter11 = np.amax(tmpMatchedFilter11) - sig11TranslationConst
		matchedFilter12 = np.amax(tmpMatchedFilter12) - sig12TranslationConst
		matchedFilter13 = np.amax(tmpMatchedFilter13) - sig13TranslationConst
		matchedFilter14 = np.amax(tmpMatchedFilter14) - sig14TranslationConst
		matchedFilter15 = np.amax(tmpMatchedFilter15) - sig15TranslationConst
		matchedFilter16 = np.amax(tmpMatchedFilter16) - sig16TranslationConst
		matchedFilter17 = np.amax(tmpMatchedFilter17) - sig17TranslationConst
		
		matchedFilter1x, matchedFilter1y = unravel_index(np.argmax(tmpMatchedFilter1), shape(tmpMatchedFilter1))
		matchedFilter2x, matchedFilter2y = unravel_index(np.argmax(tmpMatchedFilter2), shape(tmpMatchedFilter2))
		matchedFilter3x, matchedFilter3y = unravel_index(np.argmax(tmpMatchedFilter3), shape(tmpMatchedFilter3))
		matchedFilter4x, matchedFilter4y = unravel_index(np.argmax(tmpMatchedFilter4), shape(tmpMatchedFilter4))
		matchedFilter5x, matchedFilter5y = unravel_index(np.argmax(tmpMatchedFilter5), shape(tmpMatchedFilter5))
		matchedFilter6x, matchedFilter6y = unravel_index(np.argmax(tmpMatchedFilter6), shape(tmpMatchedFilter6))
		matchedFilter7x, matchedFilter7y = unravel_index(np.argmax(tmpMatchedFilter7), shape(tmpMatchedFilter7))
		matchedFilter8x, matchedFilter8y = unravel_index(np.argmax(tmpMatchedFilter8), shape(tmpMatchedFilter8))
		matchedFilter9x, matchedFilter9y = unravel_index(np.argmax(tmpMatchedFilter9), shape(tmpMatchedFilter9))
		matchedFilter10x, matchedFilter10y = unravel_index(np.argmax(tmpMatchedFilter10), shape(tmpMatchedFilter10))
		matchedFilter11x, matchedFilter11y = unravel_index(np.argmax(tmpMatchedFilter11), shape(tmpMatchedFilter11))
		matchedFilter12x, matchedFilter12y = unravel_index(np.argmax(tmpMatchedFilter12), shape(tmpMatchedFilter12))
		matchedFilter13x, matchedFilter13y = unravel_index(np.argmax(tmpMatchedFilter13), shape(tmpMatchedFilter13))
		matchedFilter14x, matchedFilter14y = unravel_index(np.argmax(tmpMatchedFilter14), shape(tmpMatchedFilter14))
		matchedFilter15x, matchedFilter15y = unravel_index(np.argmax(tmpMatchedFilter15), shape(tmpMatchedFilter15))
		matchedFilter16x, matchedFilter16y = unravel_index(np.argmax(tmpMatchedFilter16), shape(tmpMatchedFilter16))
		matchedFilter17x, matchedFilter17y = unravel_index(np.argmax(tmpMatchedFilter17), shape(tmpMatchedFilter17))
		
		matchedFilter1Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig1, mode='same'))
		matchedFilter2Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig2, mode='same'))
		matchedFilter3Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig3, mode='same'))
		matchedFilter4Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig4, mode='same'))
		matchedFilter5Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig5, mode='same'))
		matchedFilter6Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig6, mode='same'))
		matchedFilter7Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig7, mode='same'))
		matchedFilter8Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig8, mode='same'))
		matchedFilter9Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig9, mode='same'))
		matchedFilter10Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig10, mode='same'))
		matchedFilter11Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig11, mode='same'))
		matchedFilter12Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig12, mode='same'))
		matchedFilter13Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig13, mode='same'))
		matchedFilter14Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig14, mode='same'))
		matchedFilter15Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig15, mode='same'))
		matchedFilter16Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig16, mode='same'))
		matchedFilter17Correlation = np.amax(sp.signal.correlate2d(normalizedSpectra, sig17, mode='same'))
	
		logClippedmatchedFilter1 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig1, mode='same')) - sig1TranslationConst
		logClippedmatchedFilter2 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig2, mode='same')) - sig2TranslationConst
		logClippedmatchedFilter3 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig3, mode='same')) - sig3TranslationConst
		logClippedmatchedFilter4 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig4, mode='same')) - sig4TranslationConst
		logClippedmatchedFilter5 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig5, mode='same')) - sig5TranslationConst
		logClippedmatchedFilter6 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig6, mode='same')) - sig6TranslationConst
		logClippedmatchedFilter7 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig7, mode='same')) - sig7TranslationConst
		logClippedmatchedFilter8 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig8, mode='same')) - sig8TranslationConst
		logClippedmatchedFilter9 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig9, mode='same')) - sig9TranslationConst
		logClippedmatchedFilter10 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig10, mode='same')) - sig10TranslationConst
		logClippedmatchedFilter11 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig11, mode='same')) - sig11TranslationConst
		logClippedmatchedFilter12 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig12, mode='same')) - sig12TranslationConst
		logClippedmatchedFilter13 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig13, mode='same')) - sig13TranslationConst
		logClippedmatchedFilter14 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig14, mode='same')) - sig14TranslationConst
		logClippedmatchedFilter15 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig15, mode='same')) - sig15TranslationConst
		logClippedmatchedFilter16 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig16, mode='same')) - sig16TranslationConst
		logClippedmatchedFilter17 = np.amax(sp.signal.convolve2d(scaledClippedSpectra, sig17, mode='same')) - sig17TranslationConst
	
		deMeanedmatchedFilter1 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig1, mode='same')) - sig1TranslationConst
		deMeanedmatchedFilter2 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig2, mode='same')) - sig2TranslationConst
		deMeanedmatchedFilter3 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig3, mode='same')) - sig3TranslationConst
		deMeanedmatchedFilter4 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig4, mode='same')) - sig4TranslationConst
		deMeanedmatchedFilter5 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig5, mode='same')) - sig5TranslationConst
		deMeanedmatchedFilter6 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig6, mode='same')) - sig6TranslationConst
		deMeanedmatchedFilter7 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig7, mode='same')) - sig7TranslationConst
		deMeanedmatchedFilter8 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig8, mode='same')) - sig8TranslationConst
		deMeanedmatchedFilter9 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig9, mode='same')) - sig9TranslationConst
		deMeanedmatchedFilter10 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig10, mode='same')) - sig10TranslationConst
		deMeanedmatchedFilter11 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig11, mode='same')) - sig11TranslationConst
		deMeanedmatchedFilter12 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig12, mode='same')) - sig12TranslationConst
		deMeanedmatchedFilter13 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig13, mode='same')) - sig13TranslationConst
		deMeanedmatchedFilter14 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig14, mode='same')) - sig14TranslationConst
		deMeanedmatchedFilter15 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig15, mode='same')) - sig15TranslationConst
		deMeanedmatchedFilter16 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig16, mode='same')) - sig16TranslationConst
		deMeanedmatchedFilter17 = np.amax(sp.signal.convolve2d(scaledDeMeanedSpectra, sig17, mode='same')) - sig17TranslationConst
	
		whitenedMatchedFilter1 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig1, mode='same')) - sig1TranslationConst
		whitenedMatchedFilter2 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig2, mode='same')) - sig2TranslationConst
		whitenedMatchedFilter3 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig3, mode='same')) - sig3TranslationConst
		whitenedMatchedFilter4 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig4, mode='same')) - sig4TranslationConst
		whitenedMatchedFilter5 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig5, mode='same')) - sig5TranslationConst
		whitenedMatchedFilter6 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig6, mode='same')) - sig6TranslationConst
		whitenedMatchedFilter7 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig7, mode='same')) - sig7TranslationConst
		whitenedMatchedFilter8 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig8, mode='same')) - sig8TranslationConst
		whitenedMatchedFilter9 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig9, mode='same')) - sig9TranslationConst
		whitenedMatchedFilter10 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig10, mode='same')) - sig10TranslationConst
		whitenedMatchedFilter11 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig11, mode='same')) - sig11TranslationConst
		whitenedMatchedFilter12 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig12, mode='same')) - sig12TranslationConst
		whitenedMatchedFilter13 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig13, mode='same')) - sig13TranslationConst
		whitenedMatchedFilter14 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig14, mode='same')) - sig14TranslationConst
		whitenedMatchedFilter15 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig15, mode='same')) - sig15TranslationConst
		whitenedMatchedFilter16 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig16, mode='same')) - sig16TranslationConst
		whitenedMatchedFilter17 = np.amax(sp.signal.convolve2d(normWhitenedSpectraSlice, sig17, mode='same')) - sig17TranslationConst
	
		maxMatchedFilter = np.amax([matchedFilter1, matchedFilter2, matchedFilter3, matchedFilter4, matchedFilter5,
						matchedFilter6, matchedFilter7, matchedFilter8, matchedFilter9, matchedFilter10,
						matchedFilter11, matchedFilter12, matchedFilter13, matchedFilter14, matchedFilter15,
						matchedFilter16, matchedFilter17])
		logClippedmaxMatchedFilter = np.amax([logClippedmatchedFilter1, logClippedmatchedFilter2, logClippedmatchedFilter3, 
							logClippedmatchedFilter4, logClippedmatchedFilter5, logClippedmatchedFilter6, 
							logClippedmatchedFilter7, logClippedmatchedFilter8, logClippedmatchedFilter9, 
							logClippedmatchedFilter10, logClippedmatchedFilter11, logClippedmatchedFilter12, 
							logClippedmatchedFilter13, logClippedmatchedFilter14, logClippedmatchedFilter15,
							logClippedmatchedFilter16, logClippedmatchedFilter17])
		deMeanedmaxMatchedFilter = np.amax([deMeanedmatchedFilter1, deMeanedmatchedFilter2, deMeanedmatchedFilter3, 
							deMeanedmatchedFilter4, deMeanedmatchedFilter5,	deMeanedmatchedFilter6, 
							deMeanedmatchedFilter7, deMeanedmatchedFilter8, deMeanedmatchedFilter9, 
							deMeanedmatchedFilter10, deMeanedmatchedFilter11, deMeanedmatchedFilter12,
							deMeanedmatchedFilter13, deMeanedmatchedFilter14, deMeanedmatchedFilter15,
							deMeanedmatchedFilter16, deMeanedmatchedFilter17])
		maxWhitenedMatchedFilter = np.amax([whitenedMatchedFilter1, whitenedMatchedFilter2, whitenedMatchedFilter3, 
							whitenedMatchedFilter4, whitenedMatchedFilter5,	whitenedMatchedFilter6, 
							whitenedMatchedFilter7, whitenedMatchedFilter8, whitenedMatchedFilter9, 
							whitenedMatchedFilter10, whitenedMatchedFilter11, whitenedMatchedFilter12, 
							whitenedMatchedFilter13, whitenedMatchedFilter14, whitenedMatchedFilter15,
							whitenedMatchedFilter16, whitenedMatchedFilter17])
		
		avgMatchedFilterX = np.mean([matchedFilter1x, matchedFilter2x, matchedFilter3x, matchedFilter4x, 
										matchedFilter5x, matchedFilter6x, matchedFilter7x, matchedFilter8x, 
										matchedFilter9x, matchedFilter10x, matchedFilter11x, matchedFilter12x, 
										matchedFilter13x, matchedFilter14x, matchedFilter15x, matchedFilter16x,
										matchedFilter17x])
		avgMatchedFilterY = np.mean([matchedFilter1y, matchedFilter2y, matchedFilter3y, matchedFilter4y, 
										matchedFilter5y, matchedFilter6y ,matchedFilter7y ,matchedFilter8y, 
										matchedFilter9y, matchedFilter10y, matchedFilter11y, matchedFilter12y, 
										matchedFilter13y, matchedFilter14y, matchedFilter15y, matchedFilter16y,
										matchedFilter17y])
	
		maxMatchedFilterCorr = np.amax([matchedFilter1Correlation, matchedFilter2Correlation, 
										matchedFilter3Correlation, matchedFilter4Correlation, 
										matchedFilter5Correlation, matchedFilter6Correlation, 
										matchedFilter7Correlation, matchedFilter8Correlation, 
										matchedFilter9Correlation, matchedFilter10Correlation, 
										matchedFilter11Correlation, matchedFilter12Correlation, 
										matchedFilter13Correlation, matchedFilter14Correlation, 
										matchedFilter15Correlation, matchedFilter16Correlation,
										matchedFilter17Correlation])
	
		#Calculate a few crude h0 filters
		matchedFilterH01 = np.amax(sp.signal.convolve2d(normalizedSpectra, h01, mode='same'))
		matchedFilterH02 = np.amax(sp.signal.convolve2d(normalizedSpectra, h02, mode='same'))
		#matchedFilterH03 = np.amax(sp.signal.convolve2d(normalizedSpectra, h03, mode='same'))
		#matchedFilterH04 = np.amax(sp.signal.convolve2d(normalizedSpectra, h04, mode='same'))
		matchedFilterH05 = np.amax(sp.signal.convolve2d(normalizedSpectra, h05, mode='same'))
		matchedFilterH06 = np.amax(sp.signal.convolve2d(normalizedSpectra, h06, mode='same'))
	
		#Calculate a number of crude features
		sliceMean = np.mean(log(spectraSlice))
		sliceVar = np.var(log(spectraSlice))
		backMean = np.mean(log(backgroundSlice))
		backVar = np.var(log(backgroundSlice))
		outMean = np.mean(log(currSpec[0])) - sliceMean
	
		sliceBackRatio = sliceMean / backMean
		sliceOutRatio = sliceMean / outMean
	
		clippedMean = np.mean(log(clippedSpectra))
		clippedVar = np.var(log(clippedSpectra))

		#Attempting to clip more noise out of the signal
		logClippedSpectra = logClippedSpectra.clip(logClipping)
	
		#Continue to calculate more crude features
		logClippedMean5 = np.mean(logClippedSpectra.clip(5))
		logClippedVar5 = np.var(logClippedSpectra.clip(5))
		logClippedMean10 = np.mean(logClippedSpectra.clip(10))
		logClippedVar10 = np.var(logClippedSpectra.clip(10))
		logClippedMean13 = np.mean(logClippedSpectra.clip(13))
		logClippedVar13 = np.var(logClippedSpectra.clip(13))
		logClippedMean15 = np.mean(logClippedSpectra.clip(15))
		logClippedVar15 = np.var(logClippedSpectra.clip(15))
	
		frequencyMeans = np.mean(logClippedSpectra, 1)
		frequencyVars  = np.var(logClippedSpectra, 1)
		frequencySkews = sp.stats.skew(normalizedSpectra, 1)
		
		timeMeans = np.mean(logClippedSpectra, 0)
		timeVars  = np.var(logClippedSpectra, 0)
		timeSkews = sp.stats.skew(normalizedSpectra, 0)
		
	
		meanAndVars = frequencyMeans.tolist() + frequencyVars.tolist() + frequencySkews.tolist() + timeMeans.tolist() + timeVars.tolist() + timeSkews.tolist()
		#meanAndVars = timeMeans.tolist() + timeVars.tolist()
	
		#Place all those crude features into our feature array
		featureArray = [maxMatchedFilter, matchedFilter1, matchedFilter2, matchedFilter3, 
						matchedFilter4, matchedFilter5, matchedFilter6, matchedFilter7, matchedFilter8,
						matchedFilter9, matchedFilter10, matchedFilter11, matchedFilter12, matchedFilter13,
						matchedFilter14,  matchedFilter15, matchedFilter16, matchedFilter17,

						matchedFilter1x, matchedFilter2x, matchedFilter3x, matchedFilter4x, matchedFilter5x, matchedFilter6x, matchedFilter7x, matchedFilter8x, matchedFilter9x, matchedFilter10x, matchedFilter11x, matchedFilter12x, matchedFilter13x, matchedFilter14x, matchedFilter15x, matchedFilter16x, matchedFilter17x,
						matchedFilter1y, matchedFilter2y, matchedFilter3y, matchedFilter4y, matchedFilter5y, matchedFilter6y ,matchedFilter7y ,matchedFilter8y, matchedFilter9y, matchedFilter10y, matchedFilter11y, matchedFilter12y, matchedFilter13y, matchedFilter14y, matchedFilter15y, matchedFilter16y, matchedFilter17y,
						avgMatchedFilterX, avgMatchedFilterY,

						maxMatchedFilterCorr,
						matchedFilter1Correlation, matchedFilter2Correlation, matchedFilter3Correlation, matchedFilter4Correlation, matchedFilter5Correlation, matchedFilter6Correlation, matchedFilter7Correlation, matchedFilter8Correlation, matchedFilter9Correlation, matchedFilter10Correlation, matchedFilter11Correlation, matchedFilter12Correlation, matchedFilter13Correlation, matchedFilter14Correlation, matchedFilter15Correlation, matchedFilter16Correlation, matchedFilter17Correlation,

						logClippedmaxMatchedFilter, logClippedmatchedFilter1,
						logClippedmatchedFilter2, logClippedmatchedFilter3, logClippedmatchedFilter4,
						logClippedmatchedFilter5, logClippedmatchedFilter6, logClippedmatchedFilter7,
						logClippedmatchedFilter8, logClippedmatchedFilter9, logClippedmatchedFilter10,
						logClippedmatchedFilter11,  logClippedmatchedFilter12, logClippedmatchedFilter13,
						logClippedmatchedFilter14, logClippedmatchedFilter15, logClippedmatchedFilter16, 
						logClippedmatchedFilter17, 

						deMeanedmaxMatchedFilter,
						deMeanedmatchedFilter1, deMeanedmatchedFilter2, deMeanedmatchedFilter3,
						deMeanedmatchedFilter4, deMeanedmatchedFilter5, deMeanedmatchedFilter6,
						deMeanedmatchedFilter7, deMeanedmatchedFilter8, deMeanedmatchedFilter9,
						deMeanedmatchedFilter10, deMeanedmatchedFilter11, deMeanedmatchedFilter12,
						deMeanedmatchedFilter13, deMeanedmatchedFilter14, deMeanedmatchedFilter15,
						deMeanedmatchedFilter16, deMeanedmatchedFilter17,

						maxWhitenedMatchedFilter, whitenedMatchedFilter1, whitenedMatchedFilter2,
						whitenedMatchedFilter3, whitenedMatchedFilter4, whitenedMatchedFilter5,
						whitenedMatchedFilter6, whitenedMatchedFilter7, whitenedMatchedFilter8, 
						whitenedMatchedFilter9, whitenedMatchedFilter10, whitenedMatchedFilter11,
						whitenedMatchedFilter12, whitenedMatchedFilter13, whitenedMatchedFilter14,
						whitenedMatchedFilter15, whitenedMatchedFilter16, whitenedMatchedFilter17,

						matchedFilterH01, matchedFilterH02, 
						matchedFilterH05, matchedFilterH06,

						sliceMean, sliceVar, backMean, 	backVar, outMean, sliceBackRatio, sliceOutRatio, 
						clippedMean, 	logClippedMean5, logClippedVar5, logClippedMean10, 
						logClippedVar10, logClippedMean13, logClippedVar13
						]

		#Place the features into the data array for classification
		#dataArray.append(featureArray)
		dataArray.append(featureArray+meanAndVars)
	
	#Now that we have our completed dataArray conver it to a dataframe and return it
	featureList = defFeatureList(len(frequencySlice), len(timeSlice))
	featureDF = pd.DataFrame(dataArray, index=clipInfoCSV["clip_name"][0:nData], columns=featureList)
	return featureDF
	
def populatePriors(priorArray, smoothWindow=1000):
	trainPriors = smooth(priorArray, window_len=smoothWindow)
	trainPriors = trainPriors[0:30000] 

	x = arange(0,30000)
	y = trainPriors
	testPriorsRange = np.linspace(0, 30000, 54503)

	testPriors = np.interp(testPriorsRange, x, y)
	
	return trainPriors, testPriors

