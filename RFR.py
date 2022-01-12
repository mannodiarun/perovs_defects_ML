from __future__ import print_function
import numpy as np    
import csv
import copy
import pandas
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared





    # Read Data

ifile  = open('Data.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)   
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0]) 
csvdata = np.array(csvdata).reshape(numrow,numcol)
MAPbX3 = csvdata[:,0]
dopant = csvdata[:,1]
CT_p2_p1 = csvdata[:,2]
CT_p1_ze = csvdata[:,3]
CT_ze_m1  = csvdata[:,4]
CT_m1_m2  = csvdata[:,5]
E_form_X_rich  = csvdata[:,6]
E_form_Pb_rich  = csvdata[:,7]
Y1 = csvdata[:,2:6]
Y2 = csvdata[:,6:8]
X = csvdata[:,8:]


    # Read Outside Data
ifile  = open('Outside.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0])
csvdata = np.array(csvdata).reshape(numrow,numcol)
MAPbX3_out = csvdata[:,0]
M_out   = csvdata[:,1]
X_out = csvdata[:,2:]

n_out = M_out.size







XX = copy.deepcopy(X)
prop1 = copy.deepcopy(CT_p2_p1)
prop2 = copy.deepcopy(CT_p1_ze)
prop3 = copy.deepcopy(CT_ze_m1)
prop4 = copy.deepcopy(CT_m1_m2)
prop5 = copy.deepcopy(E_form_X_rich)
prop6 = copy.deepcopy(E_form_Pb_rich)
n = dopant.size
m = int(X.size/n)

t = 0.20

X_train, X_test, CT1_train, CT1_test, CT2_train, CT2_test, CT3_train, CT3_test, CT4_train, CT4_test, E_form_X_rich_train, E_form_X_rich_test, E_form_Pb_rich_train, E_form_Pb_rich_test  = train_test_split(XX, prop1, prop2, prop3, prop4, prop5, prop6, test_size=t)

n_tr = CT1_train.size
n_te = CT1_test.size

X_train_fl = [[0.0 for a in range(m)] for b in range(n_tr)]
for i in range(0,n_tr):
    for j in range(0,m):
        X_train_fl[i][j] = float(X_train[i][j])

X_test_fl = [[0.0 for a in range(m)] for b in range(n_te)]
for i in range(0,n_te):
    for j in range(0,m):
        X_test_fl[i][j] = float(X_test[i][j])

X_out_fl = [[0.0 for a in range(m)] for b in range(n_out)]
for i in range(0,n_te):
    for j in range(0,m):
        X_out_fl[i][j] = float(X_out[i][j])


Pred_out_fl  =  [[0.0 for a in range(6)] for b in range(n_out)]
err_up_out   =  [[0.0 for a in range(6)] for b in range(n_out)]
err_down_out =  [[0.0 for a in range(6)] for b in range(n_out)]


feature_importances = [[0.0 for a in range(6)] for b in range(19)]













    ####      Define Random Forest Hyperparameter Space     ####


param_grid = {
"n_estimators": [100, 200, 500],
"max_features": [10, 15, m],
"min_samples_leaf": [5,10,20],
"max_depth": [5,10,15],
"min_samples_split": [2, 5, 10]
}

param_grid = {
"n_estimators": [100],
"max_features": [15],
"max_depth": [10]
}














  ##  Train +2/+1 Transition Level Model  ##


Prop_train = copy.deepcopy(CT1_train)
Prop_test  = copy.deepcopy(CT1_test)

Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))


rfreg_opt = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)

rfreg_opt.fit(X_train_fl,Prop_train_fl)
Pred_train_fl = rfreg_opt.predict(X_train_fl)
Pred_test_fl  = rfreg_opt.predict(X_test_fl)

for i in range(0,m):
    feature_importances[i][0] = rfreg_opt.best_estimator_.feature_importances_[i]

Prop_train_CT1 = copy.deepcopy(Prop_train_fl)
Pred_train_CT1 = copy.deepcopy(Pred_train_fl)
Prop_test_CT1  = copy.deepcopy(Prop_test_fl)
Pred_test_CT1  = copy.deepcopy(Pred_test_fl)


## Outside Predictions

Pred_out = rfreg_opt.predict(X_out_fl)
for i in range(0,n_out):
    Pred_out_fl[i][0] = float(Pred_out[i])















  ##  Train +1/0 Transition Level Model  ##


Prop_train = copy.deepcopy(CT2_train)
Prop_test  = copy.deepcopy(CT2_test)

Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))
    

rfreg_opt = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)

rfreg_opt.fit(X_train_fl,Prop_train_fl)
Pred_train_fl = rfreg_opt.predict(X_train_fl)
Pred_test_fl  = rfreg_opt.predict(X_test_fl)

for i in range(0,m):
    feature_importances[i][1] = rfreg_opt.best_estimator_.feature_importances_[i]

Prop_train_CT2 = copy.deepcopy(Prop_train_fl)
Pred_train_CT2 = copy.deepcopy(Pred_train_fl)
Prop_test_CT2  = copy.deepcopy(Prop_test_fl)
Pred_test_CT2  = copy.deepcopy(Pred_test_fl)


## Outside Predictions

Pred_out = rfreg_opt.predict(X_out_fl)
for i in range(0,n_out):
    Pred_out_fl[i][1] = float(Pred_out[i])















  ##  Train 0/-1 Transition Level Model  ##


Prop_train = copy.deepcopy(CT3_train)
Prop_test  = copy.deepcopy(CT3_test)

Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))
    

rfreg_opt = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)

rfreg_opt.fit(X_train_fl,Prop_train_fl)
Pred_train_fl = rfreg_opt.predict(X_train_fl)
Pred_test_fl  = rfreg_opt.predict(X_test_fl)

for i in range(0,m):
    feature_importances[i][2] = rfreg_opt.best_estimator_.feature_importances_[i]

Prop_train_CT3 = copy.deepcopy(Prop_train_fl)
Pred_train_CT3 = copy.deepcopy(Pred_train_fl)
Prop_test_CT3  = copy.deepcopy(Prop_test_fl)
Pred_test_CT3  = copy.deepcopy(Pred_test_fl)


## Outside Predictions

Pred_out = rfreg_opt.predict(X_out_fl)
for i in range(0,n_out):
    Pred_out_fl[i][2] = float(Pred_out[i])













  ##  Train -1/-2 Transition Level Model  ##


Prop_train = copy.deepcopy(CT4_train)
Prop_test  = copy.deepcopy(CT4_test)

Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))
    

rfreg_opt = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)

rfreg_opt.fit(X_train_fl,Prop_train_fl)
Pred_train_fl = rfreg_opt.predict(X_train_fl)
Pred_test_fl  = rfreg_opt.predict(X_test_fl)

for i in range(0,m):
    feature_importances[i][3] = rfreg_opt.best_estimator_.feature_importances_[i]

Prop_train_CT4 = copy.deepcopy(Prop_train_fl)
Pred_train_CT4 = copy.deepcopy(Pred_train_fl)
Prop_test_CT4  = copy.deepcopy(Prop_test_fl)
Pred_test_CT4  = copy.deepcopy(Pred_test_fl)


## Outside Predictions

Pred_out = rfreg_opt.predict(X_out_fl)
for i in range(0,n_out):
    Pred_out_fl[i][3] = float(Pred_out[i])















  ##  Train E_form_X_rich Model  ##


Prop_train = copy.deepcopy(E_form_X_rich_train)
Prop_test  = copy.deepcopy(E_form_X_rich_test)

Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))
    

rfreg_opt = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)

rfreg_opt.fit(X_train_fl,Prop_train_fl)
Pred_train_fl = rfreg_opt.predict(X_train_fl)
Pred_test_fl  = rfreg_opt.predict(X_test_fl)

for i in range(0,m):
    feature_importances[i][4] = rfreg_opt.best_estimator_.feature_importances_[i]

Prop_train_form_X_rich = copy.deepcopy(Prop_train_fl)
Pred_train_form_X_rich = copy.deepcopy(Pred_train_fl)
Prop_test_form_X_rich  = copy.deepcopy(Prop_test_fl)
Pred_test_form_X_rich  = copy.deepcopy(Pred_test_fl)


## Outside Predictions

Pred_out = rfreg_opt.predict(X_out_fl)
for i in range(0,n_out):
    Pred_out_fl[i][4] = float(Pred_out[i])












  ##  Train E_form_Pb_rich Model  ##


Prop_train = copy.deepcopy(E_form_Pb_rich_train)
Prop_test  = copy.deepcopy(E_form_Pb_rich_test)

Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))


rfreg_opt = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)

rfreg_opt.fit(X_train_fl,Prop_train_fl)
Pred_train_fl = rfreg_opt.predict(X_train_fl)
Pred_test_fl  = rfreg_opt.predict(X_test_fl)

for i in range(0,m):
    feature_importances[i][5] = rfreg_opt.best_estimator_.feature_importances_[i]

Prop_train_form_Pb_rich = copy.deepcopy(Prop_train_fl)
Pred_train_form_Pb_rich = copy.deepcopy(Pred_train_fl)
Prop_test_form_Pb_rich  = copy.deepcopy(Prop_test_fl)
Pred_test_form_Pb_rich  = copy.deepcopy(Pred_test_fl)


## Outside Predictions

Pred_out = rfreg_opt.predict(X_out_fl)
for i in range(0,n_out):
    Pred_out_fl[i][5] = float(Pred_out[i])









np.savetxt('Pred_out.csv', Pred_out_fl)

np.savetxt('feat_imp.txt', feature_importances)











mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_CT1,Pred_test_CT1)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_CT1,Pred_train_CT1)
rmse_test_CT1  = np.sqrt(mse_test_prop)
rmse_train_CT1 = np.sqrt(mse_train_prop)
print('rmse_test_CT1 = ', np.sqrt(mse_test_prop))
print('rmse_train_CT1 = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_CT2,Pred_test_CT2)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_CT2,Pred_train_CT2)
rmse_test_CT2  = np.sqrt(mse_test_prop)
rmse_train_CT2 = np.sqrt(mse_train_prop)
print('rmse_test_CT2 = ', np.sqrt(mse_test_prop))
print('rmse_train_CT2 = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_CT3,Pred_test_CT3)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_CT3,Pred_train_CT3)
rmse_test_CT3  = np.sqrt(mse_test_prop)
rmse_train_CT3 = np.sqrt(mse_train_prop)
print('rmse_test_CT3 = ', np.sqrt(mse_test_prop))
print('rmse_train_CT3 = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_CT4,Pred_test_CT4)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_CT4,Pred_train_CT4)
rmse_test_CT4  = np.sqrt(mse_test_prop)
rmse_train_CT4 = np.sqrt(mse_train_prop)
print('rmse_test_CT4 = ', np.sqrt(mse_test_prop))
print('rmse_train_CT4 = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_form_X_rich, Pred_test_form_X_rich)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_form_X_rich, Pred_train_form_X_rich)
rmse_test_form_X_rich  = np.sqrt(mse_test_prop)
rmse_train_form_X_rich = np.sqrt(mse_train_prop)
print('rmse_test_form_X_rich  = ', np.sqrt(mse_test_prop))
print('rmse_train_form_X_rich = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_form_Pb_rich, Pred_test_form_Pb_rich)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_form_Pb_rich, Pred_train_form_Pb_rich)
rmse_test_form_Pb_rich  = np.sqrt(mse_test_prop)
rmse_train_form_Pb_rich = np.sqrt(mse_train_prop)
print('rmse_test_form_Pb_rich = ', np.sqrt(mse_test_prop))
print('rmse_train_form_Pb_rich = ', np.sqrt(mse_train_prop))
print('      ')














## ML Parity Plots ##


#fig, ( [ax1, ax2], [ax3, ax4], [ax5, ax6] ) = plt.subplots( nrows=3, ncols=2, figsize=(6,6) )

#fig, ( [ax1, ax2], [ax3, ax4] ) = plt.subplots( nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8) )

fig, ( [ax1, ax2], [ax3, ax4], [ax5, ax6] ) = plt.subplots( nrows=3, ncols=2, figsize=(8,10) )

fig.text(0.5, 0.02, 'DFT Calculation', ha='center', fontsize=32)
fig.text(0.02, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=32)

#fig, axes2d = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(6,6))

plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.95, wspace=0.25, hspace=0.4)
plt.rc('font', family='Arial narrow')
#plt.tight_layout()
#plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.5)

#plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
#plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)





Prop_train_temp = copy.deepcopy(Prop_train_CT1)
Pred_train_temp = copy.deepcopy(Pred_train_CT1)
Prop_test_temp  = copy.deepcopy(Prop_test_CT1)
Pred_test_temp  = copy.deepcopy(Pred_test_CT1)

a = [-175,0,125]
b = [-175,0,125]
ax1.plot(b, a, c='k', ls='-')

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

ax1.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax1.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_CT1
tr = '%.2f' % rmse_train_CT1

ax1.text(-0.10, -1.97, 'Train_rmse = ', c='r', fontsize=12)
ax1.text(1.02, -1.97, tr, c='r', fontsize=12)
ax1.text(1.43, -1.97, 'eV', c='r', fontsize=12)
ax1.text(-0.05, -1.62, 'Test_rmse = ', c='r', fontsize=12)
ax1.text(1.02, -1.62, te, c='r', fontsize=12)
ax1.text(1.43, -1.62, 'eV', c='r', fontsize=12)

ax1.set_ylim([-2.3, 1.8])
ax1.set_xlim([-2.3, 1.8])
ax1.set_xticks([-2, -1, 0, 1])
ax1.set_yticks([-2, -1, 0, 1])

ax1.set_title('(+2/+1) Transition Level', c='k', fontsize=20, pad=10)

ax1.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})









Prop_train_temp = copy.deepcopy(Prop_train_CT2)
Pred_train_temp = copy.deepcopy(Pred_train_CT2)
Prop_test_temp  = copy.deepcopy(Prop_test_CT2)
Pred_test_temp  = copy.deepcopy(Pred_test_CT2)

a = [-175,0,125]
b = [-175,0,125]
ax2.plot(b, a, c='k', ls='-')

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)

ax2.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax2.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_CT2
tr = '%.2f' % rmse_train_CT2

ax2.text(1.05, -1.1, 'Train_rmse = ', c='r', fontsize=12)
ax2.text(2.35, -1.1, tr, c='r', fontsize=12)
ax2.text(2.82, -1.1, 'eV', c='r', fontsize=12)
ax2.text(1.10, -0.68, 'Test_rmse = ', c='r', fontsize=12)
ax2.text(2.35, -0.68, te, c='r', fontsize=12)
ax2.text(2.82, -0.68, 'eV', c='r', fontsize=12)

ax2.set_ylim([-1.5, 3.2])
ax2.set_xlim([-1.5, 3.2])
ax2.set_xticks([-1, 0.0, 1, 2, 3])
ax2.set_yticks([-1, 0.0, 1, 2, 3])

ax2.set_title('(+1/0) Transition Level', c='k', fontsize=20, pad=10)











Prop_train_temp = copy.deepcopy(Prop_train_CT3)
Pred_train_temp = copy.deepcopy(Pred_train_CT3)
Prop_test_temp  = copy.deepcopy(Prop_test_CT3)
Pred_test_temp  = copy.deepcopy(Pred_test_CT3)

a = [-175,0,125]
b = [-175,0,125]
ax3.plot(b, a, c='k', ls='-')

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

ax3.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax3.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_CT3
tr = '%.2f' % rmse_train_CT3

ax3.text(1.86, 0.22, 'Test_rmse = ', c='r', fontsize=12)
ax3.text(3.01, 0.22, te, c='r', fontsize=12)
ax3.text(3.45, 0.22, 'eV', c='r', fontsize=12)
ax3.text(1.81, -0.15, 'Train_rmse = ', c='r', fontsize=12)
ax3.text(3.01, -0.15, tr, c='r', fontsize=12)
ax3.text(3.45, -0.15, 'eV', c='r', fontsize=12)

ax3.set_ylim([-0.5, 3.8])
ax3.set_xlim([-0.5, 3.8])
ax3.set_xticks([0, 1, 2, 3])
ax3.set_yticks([0, 1, 2, 3])

ax3.set_title('(0/-1) Transition Level', c='k', fontsize=20, pad=10)











Prop_train_temp = copy.deepcopy(Prop_train_CT4)
Pred_train_temp = copy.deepcopy(Pred_train_CT4)
Prop_test_temp  = copy.deepcopy(Prop_test_CT4)
Pred_test_temp  = copy.deepcopy(Pred_test_CT4)

a = [-175,0,125]
b = [-175,0,125]
ax4.plot(b, a, c='k', ls='-')

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)

ax4.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax4.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_CT4
tr = '%.2f' % rmse_train_CT4

ax4.text(2.52, 0.92, 'Test_rmse = ', c='r', fontsize=12)
ax4.text(3.68, 0.92, te, c='r', fontsize=12)
ax4.text(4.11, 0.92, 'eV', c='r', fontsize=12)
ax4.text(2.47, 0.55, 'Train_rmse = ', c='r', fontsize=12)
ax4.text(3.68, 0.55, tr, c='r', fontsize=12)
ax4.text(4.11, 0.55, 'eV', c='r', fontsize=12)

ax4.set_ylim([0.2, 4.5])
ax4.set_xlim([0.2, 4.5])
ax4.set_xticks([1, 2, 3, 4])
ax4.set_yticks([1, 2, 3, 4])

ax4.set_title('(-1/-2) Transition Level', c='k', fontsize=20, pad=10)








Prop_train_temp = copy.deepcopy(Prop_train_form_X_rich)
Pred_train_temp = copy.deepcopy(Pred_train_form_X_rich)
Prop_test_temp  = copy.deepcopy(Prop_test_form_X_rich)
Pred_test_temp  = copy.deepcopy(Pred_test_form_X_rich)

a = [-175,0,125]
b = [-175,0,125]
ax5.plot(b, a, c='k', ls='-')

ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)

ax5.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax5.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_form_X_rich
tr = '%.2f' % rmse_train_form_X_rich

ax5.set_ylim([-1.0, 8.5])
ax5.set_xlim([-1.0, 8.5])

ax5.text(4.27, 0.65, 'Test_rmse = ', c='r', fontsize=12)
ax5.text(6.75, 0.65, te, c='r', fontsize=12)
ax5.text(7.67, 0.65, 'eV', c='r', fontsize=12)
ax5.text(4.14, -0.18, 'Train_rmse = ', c='r', fontsize=12)
ax5.text(6.75, -0.18, tr, c='r', fontsize=12)
ax5.text(7.67, -0.18, 'eV', c='r', fontsize=12)

ax5.set_xticks([0, 2, 4, 6, 8])
ax5.set_yticks([0, 2, 4, 6, 8])

ax5.set_title('Formation Energy (X-rich)', c='k', fontsize=20, pad=10)










Prop_train_temp = copy.deepcopy(Prop_train_form_Pb_rich)
Pred_train_temp = copy.deepcopy(Pred_train_form_Pb_rich)
Prop_test_temp  = copy.deepcopy(Prop_test_form_Pb_rich)
Pred_test_temp  = copy.deepcopy(Pred_test_form_Pb_rich)

a = [-175,0,125]
b = [-175,0,125]
ax6.plot(b, a, c='k', ls='-')

ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)

ax6.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax6.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_form_Pb_rich
tr = '%.2f' % rmse_train_form_Pb_rich

ax6.set_ylim([-1.0, 9.0])
ax6.set_xlim([-1.0, 9.0])

ax6.text(4.60, 0.75, 'Test_rmse = ', c='r', fontsize=12)
ax6.text(7.2, 0.75, te, c='r', fontsize=12)
ax6.text(8.15, 0.75, 'eV', c='r', fontsize=12)
ax6.text(4.48, -0.1, 'Train_rmse = ', c='r', fontsize=12)
ax6.text(7.2, -0.1, tr, c='r', fontsize=12)
ax6.text(8.15, -0.1, 'eV', c='r', fontsize=12)

ax6.set_xticks([0, 2, 4, 6, 8])
ax6.set_yticks([0, 2, 4, 6, 8])

ax6.set_title('Formation Energy (Pb-rich)', c='k', fontsize=20, pad=10)

















#plt.tick_params(axis='y', which='both', labelleft=True, labelright=False)

#plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
#plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)

#plt.rc('xtick', c='k', labelsize=16)
#plt.rc('ytick', c='k', labelsize=24)

plt.savefig('plot.eps', dpi=450)
plt.show()






