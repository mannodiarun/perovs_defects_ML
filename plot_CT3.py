from __future__ import print_function
import numpy as np    
import csv
import copy
import random
#import mlpy
import matplotlib.pyplot as plt
#from mlpy import KernelRidge
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import pandas






Data = pandas.read_excel('Corr.xlsx', 'DFT_data')



#xx = copy.deepcopy(Data.DFT_p1_ze[:])
#yy = copy.deepcopy(Data.Pred_p1_ze[:])

xx = copy.deepcopy(Data.DFT_ze_m1[:])
yy = copy.deepcopy(Data.Pred_ze_m1[:])

#xx = copy.deepcopy(Data.DFT_E_form_X_rich[:])
#yy = copy.deepcopy(Data.Pred_E_form_X_rich[:])

#xx = copy.deepcopy(Data.DFT_E_form_Pb_rich[:])
#yy = copy.deepcopy(Data.Pred_E_form_Pb_rich[:])



error_mse = sklearn.metrics.mean_squared_error(xx,yy)
error_rmse = np.round(np.sqrt(error_mse),2)


#fig=plt.figure(figsize=(16,6),dpi=450)
plt.subplots_adjust(left=0.15, bottom=0.19, right=0.95, top=0.88)
#plt.axes.set_aspect('equal')

a = [-175,0,125]
b = [-175,0,125]

plt.rc('font', family='Arial narrow')

plt.plot(b, a, c='k', ls='-')


plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)
#plt.ylim([0.0,15.0])
#plt.xlim([0.0,15.0])


#plt.ylim([-1.5, 3.2])
#plt.xlim([-1.5, 3.2])

plt.ylim([-0.5, 4.0])
plt.xlim([-0.5, 4.0])

#plt.ylim([-1.0, 8.0])
#plt.xlim([-1.0, 8.0])

#plt.ylim([-1.0, 9.0])
#plt.xlim([-1.0, 9.0])


plt.scatter(xx[:], yy[:], c='blue', marker='s', edgecolors='dimgrey', s=50, alpha=1.0, label='Training')


#plt.scatter(Prop_test_fl[:], Pred_test[:], c='orange', marker='s', edgecolors='dimgrey', s=50, alpha=1.0, label='Test')


plt.text(1.9, 0.0, 'Error_rmse = ', c='r', fontsize=20)
plt.text(3.15, 0.0, error_rmse, c='r', fontsize=20)
plt.text(3.6, 0.0, 'eV', c='r', fontsize=20)


#plt.text(4.8, 0.8, 'RMSE = ', c='r', fontsize=20)
#plt.text(6.7, 0.8, error_rmse, c='r', fontsize=20)
#plt.text(7.4, 0.8, 'eV', c='r', fontsize=20)





#plt.text(0.0, 7.2, 'y = -1.73x + 0.15', c='r', fontsize=20)

#plt.title('Formation Energy (Pb-rich)', c='k', fontsize=32)

plt.title('(0/-1) Transition Level', c='k', fontsize=32)

#plt.xticks([-1, 0, 1, 2, 3])
#plt.yticks([-1, 0, 1, 2, 3])

plt.xticks([0, 1, 2, 3, 4])
plt.yticks([0, 1, 2, 3, 4])


#plt.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':24})
#plt.legend(bbox_to_anchor=(0.6, 0.65), frameon=False, prop={'family':'Arial narrow','size':16})
#plt.plot(b, a, c='k', ls='--')
#plt.plot(b, c, c='k', ls='--')
plt.savefig('plot_CT3.eps', dpi=450)
##plt.show()










