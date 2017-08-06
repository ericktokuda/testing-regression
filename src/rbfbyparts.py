#!/usr/bin/env python3
"""Regression by parts
"""

import numpy as np
from numpy import random
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn import svm, preprocessing
import pandas
import time
from itertools import product
from scipy.interpolate import griddata
import os.path
import pickle

import matplotlib.pyplot as plt


def load_data(fullcsv):
    rawdata0 = pd.read_csv(fullcsv)

    # Filering tails of the histogram in the hour plot
    rawdata = rawdata0[(rawdata0['time'] > 7) & (rawdata0['time'] <= 19) ]

    rawmins = np.min(rawdata)[['dx', 'dy', 'ddays', 'time']]
    rawmaxs = np.max(rawdata)[['dx', 'dy', 'ddays', 'time']]

    # Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    ndata = rawdata.copy()
    np_scaled = min_max_scaler.fit_transform(ndata[['dx', 'dy', 'ddays', 'time', 'dow']])
    ndata[['dx', 'dy', 'ddays', 'time', 'dow']] = np_scaled
    return ndata, rawmins, rawmaxs

def get_traintest_data(ndata, trainratio, outdir='/tmp'):
    filename = os.path.join(outdir, 'indices.csv')
    if os.path.exists(filename):
        msk = (np.loadtxt(filename)).astype(bool)

    else:
        msk = np.random.rand(len(ndata)) < trainratio
        np.savetxt(filename, msk, fmt='%5i', delimiter='\n')
    return ndata[msk], ndata[~msk]

def main():
    fullcsv = '20170709-pedestrians.csv'
    trainratio = 0.8
    radx = 1000.0
    rady = 1000.0
    raddays = 5.0
    radtime = 4.0
    outdir = '/tmp/'

    ndata, rawmins, rawmaxs = load_data(fullcsv)
    traindata, testdata = get_traintest_data(ndata, trainratio)

    # Here I compute the suppport
    scaledddx = radx/(rawmaxs['dx'] - rawmins['dx'])
    scaledddy = rady/(rawmaxs['dy'] - rawmins['dy'])
    scaledddays = raddays/326.0
    scaleddtime = radtime/(rawmaxs['time'] - rawmins['time'])

    mins = np.min(ndata)[['dx', 'dy', 'ddays', 'time']].as_matrix()
    maxs = np.max(ndata)[['dx', 'dy', 'ddays', 'time']].as_matrix()
    steps = [scaledddx, scaledddy, scaledddays, scaleddtime]

    mygrid = list(product(*[np.arange(i, j, k)[:-1] for i,j,k in zip(mins, maxs, steps)]))

    gridfile = os.path.join(outdir, 'grid.csv')
    if not os.path.exists(gridfile):
        np.savetxt(gridfile, np.array(mygrid), delimiter=',',
                   newline='\n')

    acc = 0
    gridid = 0

    for p in mygrid:
        filtered = traindata.copy()
        filtered = filtered[(filtered['dx'] < p[0] + scaledddx) & (filtered['dx'] > p[0] - scaledddx)]
        filtered = filtered[(filtered['dy'] < p[1] + scaledddy) & (filtered['dy'] > p[1] - scaledddy)]
        filtered = filtered[(filtered['ddays'] < p[2] + scaledddays) & (filtered['ddays'] > p[2] - scaledddays)]
        filtered = filtered[(filtered['time'] < p[3] + scaleddtime) & (filtered['time'] > p[3] - scaleddtime)]

        sz = len(filtered.index)
        if sz < 4: continue

        acc += 1        

        t0 = time.time()
        skpoints = filtered[['dx', 'dy', 'ddays', 'time']]
        clf = svm.SVR(kernel='rbf', C=1.0, verbose=True, cache_size=20000)
        clf.fit(skpoints, np.array(filtered[['people']]).ravel() )
        filenamesuf = os.path.join(outdir, str(gridid))
        pickle.dump(clf, open(filenamesuf + '.pkl', 'wb'))
        filtered.to_csv(filenamesuf + '.csv')
        gridid += 1
        #print('{},{}'.format())


if __name__ == "__main__":
    main()
