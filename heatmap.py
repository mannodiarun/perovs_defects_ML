import numpy as np    
import csv
import copy
import random
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Read Data

ifile  = open('Pearson_corr.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)   
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0]) 
csvdata = np.array(csvdata).reshape(numrow,numcol)
data = csvdata[:,:]


m = 24

Corr    =  [[0.0 for a in range(m)] for b in range(6)]

Labels  =  pandas.read_excel('Corr.xlsx', 'Label')
#Prop = ['+2/+1', '+1/0', '0/-1', '-1/-2']
Prop = ['$\Delta$H(Pb-rich)', '$\Delta$H(X-rich)', '+2/+1', '+1/0', '0/-1', '-1/-2']

x = np.arange(m)
xx = [0.0]*m
for i in range(0,m):
    xx[i] = x[i]+0.5
y = [0.5,1.5,2.5,3.5,4.5,5.5]
f = 16
r = 90

for i in range(0,m):
    for j in range(0,6):
        Corr[j][i]  = np.abs( np.float16(data[j,i]) )

scale = ['linear']
plotposition = [131, 132, 133]

fig=plt.figure(figsize=(12,6),dpi=450)
plt.rcParams.update({'font.size': 16})
plt.rc('font', family='Arial narrow')
plt.subplots_adjust(left=0.14, right=1.04, top=0.96, bottom=0.45, wspace=0.2, hspace=0.2)

ax = plt.plot(plotposition[0])
plt.plot(plotposition[0])
plt.xscale(scale[0])
plt.yscale(scale[0])
plt.xlim([0,m])
plt.ylim([0,6])
plt.xticks(xx[:],Labels.Label[:],rotation=r,fontsize=20)
plt.yticks(y[:],Prop[:],fontsize=24)
#plt.rc('xtick', labelsize=14)
#plt.rc('ytick', labelsize=14)
#plt.xlabel('', fontname='Arial narrow',size=15)
#plt.ylabel('Computed Defect Property (eV)', fontname='Arial narrow',size=24)
#plt.title(plotheading[0], fontname='Arial narrow', size=16, horizontalalignment='center')
#plt.pcolor(Corr, cmap='jet')	
plt.pcolor(Corr, cmap='Greys')
#v1 = np.linspace(-0.50, 0.50, 8, endpoint=True)
#cbar = plt.colorbar(ticks=[-0.75, -0.50, 0.00, 0.50, 0.75], spacing='uniform', orientation='vertical')
#cbar.set_label(label='Correlation Coefficient', size=16)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(label='Correlation Coefficient', size=22)
#cbar.set_label(ticks=[-0.60, -0.45, -0.30, -0.15, 0.00, 0.15, 0.30, 0.45, 0.60], size=16)
#plt.colorbar()

plt.savefig('map.pdf', dpi=450)


