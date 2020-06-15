# import packages
import matplotlib.pyplot as plt
import csv
import math
import numpy as np
from scipy import stats

# close existing plots
plt.close('all')

def getRawData(fileName, delim): # extract raw data from csv file

    rawData = []
    with open(fileName, 'r') as f:
        CSVReader = csv.reader(f, delimiter = delim, skipinitialspace = True)
        array_size = len(next(CSVReader))
        rawData = [ [] for i in range(array_size)]
        for row in CSVReader:
            for (i,val) in enumerate(row):
                rawData[i].append(float(val) )
    return rawData

# user-defined parameters
dirName = '/home/chaitanya/projects/magpie/simulations/2_component_KKS/1d_KKS' # directory containing csv file

plt.figure(figsize=(8,6))
plt.rc('font', family='serif')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#
# test_data_file = dirName + '/test/test.csv'
# testData = getRawData(test_data_file, ',')
# plt.plot(testData[0],testData[1],'r-*',linewidth=2)
#
#
# const_icData_file = dirName + '/const_ic/const_ic.csv'
# const_icData = getRawData(const_icData_file, ',')
# plt.plot(const_icData[0],const_icData[1],'b-.',linewidth=2)
#
# nn_uoData_file = dirName + '/nn_uo/nn_uo.csv'
# nn_uoData = getRawData(nn_uoData_file, ',')
# plt.plot(nn_uoData[0],nn_uoData[1],'k-x',linewidth=2)
#
#
#
# plt.xlabel('Simulation time (s)',fontsize=16)
# plt.ylabel('Wall time (s)',fontsize=16)
# plt.title('Execution speed for 1D KKS simulation',fontsize=18,fontweight='bold')
# plt.legend(['Circle IC','Constant IC','NeuralNetworkIC'],fontsize=12,loc='upper right')
#
# plt.ylim(top=200.0)

training_data = getRawData(dirName + '/test/training_rate.csv',',')
plt.plot(training_data[0],training_data[1],'r',linewidth=2)
plt.title('Neural network Mean Square Error(MSE) vs Epochs',fontsize=18,fontweight='bold')
plt.ylabel('Mean Square Error (MSE) ',fontsize=16)
plt.xlabel('Number of epochs ',fontsize=16)
plt.yscale('log')
plt.xscale('log')

plt.show()
