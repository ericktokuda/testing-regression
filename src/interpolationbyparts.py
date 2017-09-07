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
from scipy.spatial.distance import cdist
import logging
import argparse

import matplotlib.pyplot as plt



def load_data(fullcsv):
    rawdata0 = pd.read_csv(fullcsv)

    # Filtering tails of the histogram in the hour plot
    rawdata = rawdata0[(rawdata0['time'] > 7) & (rawdata0['time'] <= 19) ]

    rawmins = np.min(rawdata)[['dx', 'dy', 'ddays', 'time']]
    rawmaxs = np.max(rawdata)[['dx', 'dy', 'ddays', 'time']]

    print('Data loaded')
    return rawdata, rawmins, rawmaxs

##########################################################
def get_traintest_data(ndata, trainratio, outdir='/tmp'):
    filename = os.path.join(outdir, 'indices.csv')
    if os.path.exists(filename):
        msk = (np.loadtxt(filename)).astype(bool)
    else:
        msk = np.random.rand(len(ndata)) < trainratio
        np.savetxt(filename, msk, fmt='%5i', delimiter='\n')
    print('Train-test data splitted')
    return ndata[msk], ndata[~msk]

##########################################################
def compute_idw(knownX, knownY, unknownX, eqfactor):
    _knownX = knownX.copy()
    _knownX[[0,1]] = _knownX[[0,1]] * eqfactor
    _unknownX = unknownX.copy()
    _unknownX[[0,1]] = _unknownX[[0,1]] * eqfactor
    dist = cdist(_knownX, [_unknownX])
    weights = 1.0 / dist
    weights /= weights.sum(axis=0)
    zi = np.dot(weights.T, knownY)
    return zi


##########################################################
def main():
    parser = argparse.ArgumentParser(description='Interpolate by parts')
    parser.add_argument('startratio')
    parser.add_argument('endratio')
    args = parser.parse_args()

    fullcsv = 'data/20170709-pedestrians_sample.csv'
    outdir = '/home/grace/temp/20170903-pedsinterp/'
    trainratio = 0.8
    radx = 1000.0
    rady = 1000.0
    raddays = 0
    radtime = 4.0
    deltax = radx / 2
    deltay = rady / 2
    deltadays = 1.0
    deltatime = 1.0
    # Here I am saying that a point at exact the same location
    #1h before or after has the same weight as one point at the same
    #time but 1000 away  (1h/1000)
    eqfactor = 1.0 / 1000.0  
    scaledddx = radx * eqfactor
    scaledddy = rady * eqfactor
    scaledddays = raddays / 326.0
    scaleddtime = 1
    MINPOINTS = 4
    BUFFERSZ = 10

    if not os.path.exists(outdir): os.makedirs(outdir)
    rawdata, mins, maxs = load_data(fullcsv)
    traindata, testdata = get_traintest_data(rawdata, trainratio)

    steps = [deltax, deltay, deltadays, deltatime]

    gridfile = os.path.join(outdir, '20170820-interpolationgrid.npy')
    #x = pd.DataFrame(mygrid)

    if not os.path.exists(gridfile):
        mygrid = np.array(list(product(*[np.arange(i, j, k)[:-1] for i,j,k in zip(mins, maxs, steps)])))
        np.save(gridfile, np.array(mygrid))
    else:
        mygrid = np.load(gridfile)
    print('Grid loaded')


    mygrid = (pd.DataFrame(mygrid, columns=['dx','dy','day','time']))

    acc = 0
    bufferedpoints = {}
    bufferedvalues = {}

    indices = mygrid.index
    nindices = len(indices)
    startidx = int(nindices * float(args.startratio))
    endidx = int(nindices * float(args.endratio))

    if startidx < endidx:
        deltaidx = 1
        endidx -= 1
    else:
        deltaidx = -1
        startidx -= 1


    for jj in range(startidx, endidx, deltaidx):
        idx = indices[jj]
        p = mygrid.iloc[idx]
        filtered = traindata[(traindata['dx'] < p['dx'] + radx) & (traindata['dx'] > p['dx'] - radx)].copy()
        filtered = filtered[(filtered['dy'] < p['dy'] + rady) & (filtered['dy'] > p['dy'] - rady)]
        filtered = filtered[filtered['ddays'] == p['day']]
        filtered = filtered[(filtered['time'] < p['time'] + radtime) & (filtered['time'] > p['time'] - radtime)]

        if len(filtered.index) < MINPOINTS: continue
        acc += 1        

        t0 = time.time()
        skpoints = filtered[['dx', 'dy', 'time']].as_matrix()

        filenamesuf = os.path.join(outdir, args.startratio + '_' + str(acc//BUFFERSZ))
        interpolated = compute_idw(skpoints, filtered['people'], p[1:], eqfactor)
        bufferedvalues[idx] = interpolated
        bufferedpoints[idx] = (filtered['id']).as_matrix()
        
        if acc % BUFFERSZ == BUFFERSZ-1:
            pickle.dump(bufferedvalues, open('{}_values.pkl'.format(filenamesuf), "wb"))
            pickle.dump(bufferedpoints, open('{}_points.pkl'.format(filenamesuf), "wb"))
            bufferedvalues = {}
            bufferedpoints = {}
            print(acc)

if __name__ == "__main__":
    main()
