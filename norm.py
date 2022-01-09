import numpy as np    
import csv
import copy
import random
import matplotlib.pyplot as plt
import pandas
#from sklearn import cross_validation
from sklearn.preprocessing import normalize

    # Read Data
ifile  = open('Outside.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)   
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0]) 
csvdata = np.array(csvdata).reshape(numrow,numcol)
systems = csvdata[:,0]
X = csvdata[:,2:]

n = np.int(systems.size)
m = np.int(X.size/n)

X_fl = [[0.0 for a in range(m)] for b in range(n)]
for i in range(0,n):
    for j in range(0,m):
        X_fl[i][j] = np.float(X[i][j])

#xx = [0.0]*n
#Y = [[0.0 for a in range(m)] for b in range(n)]
#for i in range(0,m):
#    for k in range(0,n):
#        xx[k] = X_fl[k][i]
#    yy = normalize(xx, norm='l2', axis=1)
#   for j in range(0,n):
#       Y[j][i] = copy.deepcopy(yy[0][j])

Y = normalize(X_fl, norm='l2', axis=0)

np.savetxt('X_norm.txt', Y)


