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

signalCount = 0
backCount   = 0

signalModel         = zeros((len(frequencySlice),len(timeSlice)))
signalModelDeMeaned = zeros((len(frequencySlice),len(timeSlice)))
#backModel = zeros((len(frequencySlice),len(timeSlice)))

ensembledPowerSpectra = np.genfromtxt(os.path.join(baseDir, "powerSpectra", "EnsembledSpectra_Iter1.csv"), delimiter=",")

#Iter 1 missed detections
#writeSigList = [28,31,32,79,180,651,673,730,795,931,1059,1101,1376,2166,2179,2234,
						 #2689,3032,3147,3465,3541,3560,3998,4009,4696,4848,5008,5009,5022,
						 #5209,5225,5237,5240,5251,5262,5264,5268,5273,5275,5278,5296,5302,
						 #5309,5315,5317,5319,5330,5367,5371,5372,5867,5950,6178,6276,6298,
						 #6316,6323,6331,6336,6346,6353,6362,6371,6387,6388,6391,6418,6426,
						 #6430,6436,6453,6472,6478,6490,6505,6508,6521,6523,6538,6567,6583,
						 #6586,6634,6635,6643,6649,6662,6672,6684,6687,6690,6696,6697,6698,
						 #6712,6732,6734,6737,6745,6751,6762,7117,7198,7207,7246,7308,7387,
						 #7589,7590,7591,7615,7631,7636,7638,7645,7683,7698,7753,7794,7797,
						 #7889,7905,8043,8060,8126,8181,8218,8219,8224,8461,8462,8555,8570,
						 #8592,8620,8631,8639,8779,8801,8808,8890,8950,9004,9021,9026,9028,
						 #9029,9057,9061,9066,9097,9148,9182,9185,9245,9304,9313,9326,9327,
						 #9330,9335,9479,9484,9485,9492,9564,9566,9648,9652,9663,9714,9723,
						 #9858,9864,9868,9923,9928,9935,9966,9981,9986,10082,10110,10121,
						 #10155,10227,10233,10239,10250,10268,10288,10380,10387,10390,10400,
						 #10423,10424,10610,10646,10704,10749,10841,10843,10891,11042,11048,
						 #11052,11165,11179,11182,11183,11201,11221,11276,11335,11428,11429,
						 #11526,11612,11622,11652,11659,11683,11703,11718,11790,11874,11923,
						 #11926,11942,11958,11981,11998,12125,12166,12167,12169,12174,12175,
						 #12214,12218,12238,12309,12319,12326,12344,12386,12573,12592,12639,
						 #12690,12693,12702,12705,12724,12914,13049,13079,13239,13337,13997,
						 #14002,16124,16159,16801,17236,17319,17419,17597,17976,18150,18250,
						 #18605,18713,18805,18847,18859,18864,18869,18969,19133,19261,19414,
						 #19447,19454,19480,19509,19510,19570,19772,19852,20354,20390,20396,
						 #20458,20477,20543,20651,20657,20834,20911,20927,21734,21803,21817,
						 #22286,22343,22421,22574,22610,22631,22853,23000,23006,23107,23108,
						 #23133,23227,23282,24430,24525,24559,24582,24605,24622,24627,24632,
						 #24947,24954,25004,25109,25145,25158,25232,25299,25613,25716,25761,
						 #25807,25817,26017,26025,26277,26282,26456,26571,26578,26611,26614,
						 #26881,26885,27076,28555,28573,28637,29478,29549,29550,29551]
						 
#Iter 1 false positives
# writeSigList = [76,964,1084,1113,1368,2211,2235,4738,5007,5071,5320,5325,5327,5354,5448,
						# 5455,5541,5562,5628,5652,5653,5666,5721,5738,5741,5925,5971,6004,6017,6018,
						# 6020,6025,6031,6032,6045,6046,6053,6079,6103,6215,6218,6233,6234,6258,6265,
						# 6327,6543,6547,6609,6665,6666,6667,6678,6863,6931,6940,6992,6999,7002,7023,
						# 7033,7149,7164,7189,7196,7229,7230,7231,7234,7236,7238,7243,7254,7274,7283,
						# 7287,7355,7359,7361,7366,7404,7407,7437,7447,7469,7482,7483,7685,7746,7758,
						# 7790,7801,7806,7860,7896,7943,7946,8039,8042,8073,8125,8152,8178,8194,8207,
						# 8209,8252,8279,8289,8290,8307,8311,8324,8335,8336,8349,8363,8366,8405,8437,
						# 8486,8512,8549,8585,8591,8623,8640,8656,8659,8674,8686,8695,8701,8705,8735,
						# 8743,8751,8757,8759,8790,8818,8828,8849,8864,8873,8910,8927,8930,8957,9002,
						# 9011,9058,9074,9084,9093,9102,9113,9140,9157,9159,9168,9172,9195,9199,9214,
						# 9216,9251,9277,9282,9292,9305,9322,9339,9389,9404,9513,9579,9611,9624,9643,
						# 9664,9669,9689,9697,9718,9769,9787,9905,9918,9944,10014,10043,10047,10075,
						# 10089,10109,10182,10195,10196,10338,10355,10437,10449,10488,10538,10556,
						# 10606,10618,10619,10716,10754,10801,10864,10956,11040,11116,11141,11144,
						# 11146,11162,11202,11226,11326,11477,11487,11511,11588,11660,11670,11676,
						# 11677,11717,11749,11754,11779,11798,11802,11878,11883,11892,11944,11957,
						# 12013,12021,12035,12036,12048,12071,12073,12216,12246,12249,12277,12313,
						# 12316,12317,12329,12343,12370,12374,12501,12531,12605,12612,12673,12860,
						# 12921,12935,12940,12975,13059,13128,13178,13310,15274,15688,17166,17275,
						# 17422,17589,17593,17706,17785,17831,17868,18066,18123,18151,18157,18171,
						# 18415,18607,18638,18972,18994,19007,19156,19198,19212,19226,19440,19694,
						# 19752,19837,19912,20007,20128,20253,20298,20346,20457,20539,20642,20659,
						# 21645,21864,21892,22136,22524,22590,22621,22826,23007,23033,23246,23276,
						# 23494,23539,23557,24442,24577,24611,24648,24962,25070,25144,25379,25768,
						# 26212,26265,26515,26540,26751,26905,27394,27396,27872,27929,27978,28249,
						# 28388,28488,28549,29929,29933]
				
#Iter 3 missed detections
writeSigList = [28573,23133,3032,27076,795,25158,25716,673,8224,23108,3465,13997,29549,
                6712,29550,10610,22853,16159,24605,24525,20354,12639,24430,14002,1101,
                8779,10891,11048,31,6323,18969,4841,4848,20543,19764,5240,9304,6391,12174,
                20390,6353,26277,1127,6307,25807,25004,5372,7683,6752,9479,12175,12386,
                5273,29478,32,11052,24469,8027,9148,5278,6380,5251,20396,5275,6734,22604,
                6505,7645,12218,5315,9923,12319,18995,6567,26017,19852,6336,9648,13079,
                10170,28,8218,23000,20476]
                
#Iter 3 FAs
#writeSigList = [8549,8279,18415,6018,9157,7254,9011,6999,27929,7860,5653,27396,8591,7359,
                #6234,7236,8674,11487,9669,27978,12673,19156,6242,5625,7746,12935,26316,17593,
                #11660,12343,7231,8336,8930,27394,9093,11144,7407,8042,7274,8828,12048,22826,
                #25768,8957,11588,9339,22136,4738,11314,8178,11676,29933,9074,20128,8252,28488,
                #29929,9697,13310,8207,6233,8125,12021,10195,26751 ]						 

writeSigList = ["train"+str(a)+".aiff" for a in writeSigList]		

print(writeSigList)				 
						 
for x in range(0, nData):
    if trainTruth["clip_name"][x] in writeSigList:
        #if trainTruth["label"][x] == 0:
        #set the filename based on the trainTruth clip_name
        currFile = os.path.join(baseDir, "train", trainTruth["clip_name"][x])

        #read in the current file
        reader = aifc.open(currFile, 'r')
        nframes = reader.getnframes()

        strsig = reader.readframes(nframes)
        currData = np.fromstring(strsig, np.short).byteswap()

        #attempt to whiten the data
        testFFT = np.fft.rfft(currData)
        testFFT = testFFT / ensembledPowerSpectra
        whiteData = np.fft.irfft(testFFT)

        #Calculate the spectrogram of the audio
        currSpec = specgram(currData)
        whiteSpec = specgram(whiteData)

        #Pull out the magnitude of the time/frequency slice where the signal resides
        # as well as a background slice in the frequency slice before and after the signal
        spectraSlice = currSpec[0][frequencySlice,:][:,timeSlice]
        backgroundSlice = currSpec[0][frequencySlice,:][:,(0,1,2,3,26,27,28,29)]
        
        whiteSlice = whiteSpec[0][frequencySlice,:][:,timeSlice]

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

        #aggregate the truth data to create a signal model
        #signalModel += spectraSlice
        #signalModelDeMeaned += clippedSpectra
        signalCount = signalCount + 1
        print "Signal count: %d"%signalCount

        #Draw the original signal spectrogram, unaltered
        ax1 = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=2)
        plt.title("Full Spectrogram")
        ax1.axes.get_xaxis().set_visible(False)
        specgram(currData)
        #draw()
        #savefig(os.path.join(baseDir, "images", "signal",  trainTruth["clip_name"][x] + ".png"))

        #Draw the signal spectrogram slice
        ax2 = plt.subplot2grid((3,3), (0,2))
        plt.title("Signal Slice")
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        plt.imshow(log(spectraSlice))
        plt.gca().invert_yaxis()

        #Clip the deMeanedSpectra so that negative numbers dont' show up as NaNs
        deMeanedSpectra = deMeanedSpectra.clip(1)

        #Draw the demeaned signal spectrogram slice
        ax3 = plt.subplot2grid((3,3), (1,2))
        plt.title("Whitened Slice")
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        plt.imshow(log(whiteSlice))
        plt.gca().invert_yaxis()

        #Draw the demeaned signal spectrogram slice, log clipped at 5
        ax4 = plt.subplot2grid((3,3), (2,0))
        plt.title("De-meaned spectra")
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)
        plt.imshow(log(deMeanedSpectra))
        plt.gca().invert_yaxis()

        #Draw the demeaned signal spectrogram slice, log clipped at 5
        ax5 = plt.subplot2grid((3,3), (2,1))
        plt.title("Log Clipped 5")
        ax5.axes.get_xaxis().set_visible(False)
        ax5.axes.get_yaxis().set_visible(False)
        plt.imshow(log(deMeanedSpectra).clip(5))
        plt.gca().invert_yaxis()

        #Draw the demeaned signal spectrogram slice, log clipped at 10
        ax6 = plt.subplot2grid((3,3), (2,2))
        plt.title("Log Clipped 10")
        ax6.axes.get_xaxis().set_visible(False)
        ax6.axes.get_yaxis().set_visible(False)
        plt.imshow(log(deMeanedSpectra).clip(10))
        plt.gca().invert_yaxis()

        # #Draw the 2d fft of the signal
        # ax7 = plt.subplot2grid((4,3), (3,0))
        # plt.title("2d fft of signal")
        # ax7.axes.get_xaxis().set_visible(False)
        # ax7.axes.get_yaxis().set_visible(False)
        # tmp = abs(np.fft.fft2(log(spectraSlice)))
        # tmp[0,0] = 0
        # plt.imshow(tmp)
        # plt.gca().invert_yaxis()

        # #Draw the 2d fft of the demeaned signal
        # ax8 = plt.subplot2grid((4,3), (3,1))
        # plt.title("2d fft of demeaned signal")
        # ax8.axes.get_xaxis().set_visible(False)
        # ax8.axes.get_yaxis().set_visible(False)
        # tmp = abs(np.fft.fft2(log(deMeanedSpectra)))
        # tmp[0,0] = 0
        # plt.imshow(tmp)
        # plt.gca().invert_yaxis()

        draw()
        #savefig(os.path.join(baseDir, "images", "Iter1FalsePositives",  trainTruth["clip_name"][x] + ".png"))
        #savefig(os.path.join(baseDir, "images", "backGrid",  trainTruth["clip_name"][x] + ".png"))
        savefig(os.path.join(baseDir, "Images", "Iter3MissedDetections",  trainTruth["clip_name"][x] + ".png"))
        #savefig(os.path.join(baseDir, "Images", "Iter3FAs",  trainTruth["clip_name"][x] + ".png"))
		
# signalModel /= signalCount
# signalModelDeMeaned /= signalCount
# #backModel /= backCount

# #Write the ensembled signal
# ax7 = plt.subplot2grid((4,3), (0,0), colspan=3, rowspan=3)
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

