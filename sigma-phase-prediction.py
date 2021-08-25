#!/usr/bin/env python
# coding: utf-8

# # sigma phase prediction from DFT
# 
# This notebook is a supplementary part of the document:
# https://arxiv.org/abs/2011.10883
# 
# Jean-Claude Crivello, Nataliya Sokolovska, Jean-Marc Joubert
# 
# crivello@icmpe.cnrs.fr

# ## (1) Preparation

# In[1]:


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[2]:


# The database was build from DFT calculations made under same conditions

database = pd.read_csv('DB-train', sep=" ", header=None)

database.columns = ['num','X1', 'X2', 'X3', 'X4', 'X5', 'F', 'a', 'c', 'ratio', 'vol', 
              'x_4f', 'x_8i1', 'y_8i1', 'x_8i2', 'y_8i2', 'x_8j', 'z_8j', 'mag']

# Only 14 elememts are considered
elements = ["Al","Co","Cr","Fe","Mn","Mo","Nb","Ni","Pt","Re","Ru","V","W","Zr"]
database.query('X1 == @elements and X2 == @elements and X3 == @elements and X4 == @elements and X5 == @elements', inplace = True)


# In[3]:


# Use of the heat of formation as the predictive variable 

from module.sigma import return_heat

database.insert(7, "H", 0.0) 
database["H"] = database.apply(return_heat, axis=1)
database.head()

# num   : configuration number
# Xi    : distribution of element in the 5 sites of the sigma phase
# F     : total energy by DFT for 30 atoms (eV)
# H     : heat of formation (kJ/mol)
# a, c  : tetragonal cell parameters (Angstrom)
# ratio : c/a
# vol   : cell volume (Angstrom^3)
# x_4f ... z_8j : internal parameters of non-equivalent positions


# In[4]:


# Preparation of the supervised information

y = database['H']
X, y = database.loc[:,'X1':'X5'], database['H']

print(X.shape, y.shape)


# In[5]:


# Dummy encoding of categorical features

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

X_ohe = ohe.fit_transform(X)


# In[6]:


# Addionnal physical features

# Atomic Radius

from module.sigma import return_radius

zero_data = np.zeros(shape=X.shape)
Radius = pd.DataFrame(zero_data, columns=['R1', 'R2', 'R3', 'R4', 'R5'])
Radius = X.applymap(return_radius)

X_new1=np.append(X_ohe, Radius, axis=1)


# Valence electron numbers

from module.sigma import return_valen_el

zero_data = np.zeros(shape=X.shape)
Electron = pd.DataFrame(zero_data, columns=['E1', 'E2', 'E3', 'E4', 'E5'])
Electron = X.applymap(return_valen_el)

X_new2=np.append(X_new1, Electron, axis=1)


# In[7]:


# Normalisation 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(Radius) 
Radius.loc[:,:] = scaled_values
scaled_values = scaler.fit_transform(Electron) 
Electron.loc[:,:] = scaled_values

# fonction
def return_radius_normed(element):
    return ((return_radius(element)-124)/(160-124))
# fonction
def return_valen_el_normed(element):
    return ((return_valen_el(element)-3)/(10-3))


# In[8]:


# New Learning database

X_new3=np.append(X_ohe, Radius, axis=1)
X_new3=np.append(X_new3, Electron, axis=1)
X_new3.shape #5*14+5*2


# ## (2) Supervised learning with neural network

# In[9]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import math

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression


# The folowing architecture has been optimized:
model = MLPRegressor(random_state=1, 
                     max_iter=1000000,
                     activation='tanh',
                     alpha=0.001,
                     hidden_layer_sizes=(500,500,500), 
                     learning_rate='constant', 
                     solver='sgd')

# learning on all data
model.fit(X_new3,y)

# simple prediction
R2_train = model.score(X_new3,y)
print("R^2 on training data =", R2_train)

y_pred = model.predict(X_new3)

MAE_train = mean_absolute_error(y, y_pred)
print("MAE on test data = ", MAE_train)
MSE_train = mean_squared_error(y, y_pred)
print("MSE on test data = ", MSE_train)
RMSE_train = math.sqrt(MSE_train)
print("RMSE on test data = ", RMSE_train)


# In[10]:


x_tmp = np.linspace(min(min(y_pred), min(y)),max(max(y_pred),max(y)),100)
y_tmp = np.linspace(min(min(y_pred), min(y)),max(max(y_pred),max(y)),100)


plt.scatter(y_pred, y,c='g',label='$R^2$ = '+ 
            str(round(R2_train,2)) + ',\n RMSE ='+ 
            str(round(RMSE_train,2)))
plt.xlabel('Predicted (test data)')
plt.ylabel('Observed (test data)')
plt.plot(x_tmp,y_tmp,c='k')

plt.title('Example of prediction $\Delta_fH$ (kJ/mol) \n Neural network + Atomic Radius + Valence Electron')
plt.legend(loc=4)


# ## (3) Check prediction

# In[11]:


# fonction
def return_calcul(A,B,C,D,E):
    """ Returns DFT calculated resuls """
    config=X[(X['X1'] == A) & (X['X2'] == B) & (X['X3'] == C) & (X['X4'] == D) & (X['X5'] == E)]
    return y[config.index[0]]

#function 
def return_predic(A,B,C,D,E):
    """ Prediction of an individual configuration """
    config=pd.DataFrame([[A,B,C,D,E]])
    temp = np.append(ohe.transform(config), config.applymap(return_radius_normed), axis=1) 
    prediction = np.append(temp, config.applymap(return_valen_el_normed), axis=1) 
    return model.predict(prediction)[0]

#function of 
def return_prediction(row):
    """ prediction of a dataframe of configurations """
    config=pd.DataFrame([[row['X1'],row['X2'],row['X3'],row['X4'],row['X5']]])
    temp = np.append(ohe.transform(config), config.applymap(return_radius_normed), axis=1) 
    prediction = np.append(temp, config.applymap(return_valen_el_normed), axis=1)
    return model.predict(prediction)[0] 

A='Re'
B='Mo'
C='Re'
D='Mo'
E='Re'

print("calculated at ", return_calcul(A,B,C,D,E))
print("predicted at  ", return_predic(A,B,C,D,E) )


# In[12]:


configurations=[["Co","Co","Co","Co","Co"],
                ["Cr","Fe","Fe","Cr","Fe"],
                ["Fe","Cr","Ni","Fe","Cr"]]

for conf in configurations:
    cal=return_calcul(conf[0],conf[1],conf[2],conf[3],conf[4])
    pre=return_predic(conf[0],conf[1],conf[2],conf[3],conf[4])
    print(conf, 'calculated / predicted at ', round(cal,2), ' / ', round(pre,2), 'kJ/mol')


# ## (4) Test on new random configuration set

# In[13]:


# The test database is composed of 1001 semi-random configurations among the 537824 

database_test = pd.read_csv('DB-test', sep=" ", header=None)

database_test.columns = ['num','X1', 'X2', 'X3', 'X4', 'X5', 'F', 'a', 'c', 'ratio', 'vol', 
              'x_4f', 'x_8i1', 'y_8i1', 'x_8i2', 'y_8i2', 'x_8j', 'z_8j', 'mag']

database_test.insert(7, "H", 0.0) 
database_test["H"] = database_test.apply(return_heat, axis=1)

def level_sys(row):
    configuration=[row['X1'],row['X2'],row['X3'],row['X4'],row['X5']]
    # list of elements
    els = []
    for el in configuration:
        if el not in els:
            els.append(el)
    level = len(els)
    return level

database_test['level'] = database_test.apply(level_sys, axis=1)
database_test.sort_values(['level'],inplace=True,ascending=False)

X_test, y_test = database_test.loc[:,'X1':'X5'], database_test['H']

print(X_test.shape, y_test.shape)


# In[14]:


X_ohe_test = ohe.fit_transform(X_test)
zero_data = np.zeros(shape=X_test.shape)
Radius_test = pd.DataFrame(zero_data, columns=['R1', 'R2', 'R3', 'R4', 'R5'])
Radius_test = X_test.applymap(return_radius)
Electron_test = pd.DataFrame(zero_data, columns=['E1', 'E2', 'E3', 'E4', 'E5'])
Electron_test = X_test.applymap(return_valen_el)
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(Radius_test) 
Radius_test.loc[:,:] = scaled_values
scaled_values = scaler.fit_transform(Electron_test) 
Electron_test.loc[:,:] = scaled_values
X_new_test=np.append(X_ohe_test, Radius_test, axis=1)
X_new_test=np.append(X_new_test, Electron_test, axis=1)


# In[15]:


# R2 on test
R2_test = model.score(X_new_test, y_test)
print("R^2 on testing data =", R2_test)

# Make predictions using the testing set
y_prtra = model.predict(X_new3)
y_pred = model.predict(X_new_test)

MAE_test = mean_absolute_error(y_test, y_pred)
print("MAE on test data = ", MAE_test)
MSE_test = mean_squared_error(y_test, y_pred)
print("MSE on test data = ", MSE_test)
RMSE_test = math.sqrt(MSE_test)
print("RMSE on test data = ", RMSE_test)


# In[16]:


y_prtra_meV, y_pred_meV, y_test_meV, y_meV = y_prtra*9.6486, y_pred*9.6486, y_test*9.6486, y*9.6486
MAE_test_meV, RMSE_test_meV = MAE_test*9.6486, RMSE_test*9.6486

x_tmp = np.linspace(min(min(y_pred_meV), min(y)),max(max(y_pred_meV),max(y_meV)),100)
y_tmp = np.linspace(min(min(y_pred_meV), min(y)),max(max(y_pred_meV),max(y_meV)),100)

plt.scatter(y_prtra_meV, y_meV,c='r',label='training: R^2 = '+ str(round(R2_train,2)))
plt.scatter(y_pred_meV, y_test_meV,c='b',label='test: R^2 = '+ str(round(R2_test,2)) + '\n  MAE ='+ str(round(MAE_test_meV,2)) +', RMSE ='+ str(round(RMSE_test_meV,2)))
plt.xlabel('ML predicted $\Delta_f H$ (meV/at)')
plt.ylabel('DFT calculated $\Delta_f H$ (meV/at)')
plt.plot(x_tmp,y_tmp,c='k')

plt.title('MPR with additional physical features')
plt.legend(loc=4)
#plt.savefig('fig/fig9a.png', dpi=300)


# ## (5) Analysis of the system degree

# In[17]:


# 
# fonction qui donne la combinatoire de système de degreé k pour une liste seq
def combinliste(seq, k):
    p = []
    i, imax = 0, 2**len(seq)-1
    while i<=imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j<=jmax:
            if (i>>j)&1==1:
                s.append(seq[j])
            j += 1
        if len(s)==k:
            p.append(s)
        i += 1 
    return p

def level_sys(row):
    configuration=[row['X1'],row['X2'],row['X3'],row['X4'],row['X5']]
    # list of elements
    els = []
    for el in configuration:
        if el not in els:
            els.append(el)
    level = len(els)
    return level

def ratio_data(row):
    configuration=[row['X1'],row['X2'],row['X3'],row['X4'],row['X5']]
    # list of elements
    els = []
    for el in configuration:
        if el not in els:
            els.append(el)
    nb_els = len(els)
    # preparation
    selection = database.copy()
    selection.query('X1 == @els and X2 == @els and X3 == @els and X4 == @els and X5 == @els', inplace = True)
    nb_config_data = selection.F.count()
    nb_config_theo = nb_els**5
    return round(nb_config_data / nb_config_theo,2) 

def number_bin(row,degre_studied=2):
    configuration=[row['X1'],row['X2'],row['X3'],row['X4'],row['X5']]
    # list of the system
    system = []
    for el in configuration:
        if el not in system:
            system.append(el)
    degre = len(system)
    # preparation
    num_bin_theo = len(combinliste(system,degre_studied))
    #print(combinliste(system,degre_studied))
    num_bin = 0
    for sys in combinliste(system,degre_studied):
        selection = database.copy()
        selection.query('X1 == @sys and X2 == @sys and X3 == @sys and X4 == @sys and X5 == @sys', inplace = True)
        nb_config_data = selection.F.count()
        nb_config_theo = degre_studied**5
        #print(nb_config_data,nb_config_theo)
        if nb_config_data == nb_config_theo:
            num_bin+=1
    return round(num_bin/num_bin_theo,2)  

def number_ter(row,degre_studied=3):
    configuration=[row['X1'],row['X2'],row['X3'],row['X4'],row['X5']]
    # list of the system
    system = []
    for el in configuration:
        if el not in system:
            system.append(el)
    degre = len(system)
    # preparation
    num_ter_theo = len(combinliste(system,degre_studied))
    #print(combinliste(system,degre_studied))
    num_ter = 0
    for sys in combinliste(system,degre_studied):
        selection = database.copy()
        selection.query('X1 == @sys and X2 == @sys and X3 == @sys and X4 == @sys and X5 == @sys', inplace = True)
        nb_config_data = selection.F.count()
        nb_config_theo = degre_studied**5
        #print(nb_config_data,nb_config_theo)
        if nb_config_data == nb_config_theo:
            num_ter+=1
        if num_ter_theo == 0:
            return 0.0
        else:
            return round(num_ter/num_ter_theo,2) 


# In[18]:


# WARNING: could take several minutes

#from module.sigma import number_bin 
X_plot = X_test.copy()
X_plot['level'] = X_plot.apply(level_sys, axis=1)
X_plot['ratio'] = X_plot.apply(ratio_data, axis=1)
X_plot['Nb_bin'] = X_plot.apply(number_bin, axis=1)
X_plot['Nb_ter'] = X_plot.apply(number_ter, axis=1)
X_plot.head()


# In[21]:


import seaborn as sns
sns.set()
color_map=plt.cm.get_cmap('viridis_r', 4)

y_prtra_meV, y_pred_meV, y_test_meV, y_meV = y_prtra*9.6486, y_pred*9.6486, y_test*9.6486, y*9.6486
MAE_test_meV, RMSE_test_meV = MAE_test*9.6486, RMSE_test*9.6486

x_tmp = np.linspace(min(min(y_pred_meV), min(y)),max(max(y_pred_meV),max(y_meV)),100)
y_tmp = np.linspace(min(min(y_pred_meV), min(y)),max(max(y_pred_meV),max(y_meV)),100)


plt.xlabel('ML predicted $\Delta_f H$ (meV/at)')
plt.ylabel('DFT calculated $\Delta_f H$ (meV/at)')
plt.plot(x_tmp,y_tmp,c='k', zorder=1)

plt.scatter(y_prtra_meV, y_meV,c='red',label='training: R^2 = '+ str(round(R2_train,2)))

plt.scatter(y_pred_meV, y_test_meV,c=X_plot['level'],cmap=color_map, zorder=2,
            label='test: R^2 = '+ str(round(R2_test,2)) + '\n  MAE ='+ str(round(MAE_test_meV,2)) +', RMSE ='+ str(round(RMSE_test_meV,2)))

plt.colorbar(ticks=[2, 3, 4, 5], label='degree of the system')
plt.title('Multi-layer Perceptron Regressor (MPR)')

legend = plt.legend(loc=4)
frame = legend.get_frame()
frame.set_facecolor('white')
frameon=True


# ## (6) Final prediction on every configurations

# In[22]:


# import of all DFT data

database_1 = pd.read_csv('DB-train', sep=" ", header=None)
database_2 = pd.read_csv('DB-test', sep=" ", header=None)

frame = [ database_1, database_2 ]
database = pd.concat(frame)

database.columns = ['num','X1', 'X2', 'X3', 'X4', 'X5', 'F', 'a', 'c', 'ratio', 'vol', 
              'x_4f', 'x_8i1', 'y_8i1', 'x_8i2', 'y_8i2', 'x_8j', 'z_8j', 'mag']

elements = ["Al","Co","Cr","Fe","Mn","Mo","Nb","Ni","Pt","Re","Ru","V","W","Zr"]
database.query('X1 == @elements and X2 == @elements and X3 == @elements and X4 == @elements and X5 == @elements', inplace = True)

database.drop_duplicates(inplace=True)

database.insert(7, "H", 0.0) 
database["H"] = database.apply(return_heat, axis=1)

database.shape


# In[23]:


# add the degree of the system 

database['level'] = database.apply(level_sys, axis=1)
database.head()


# In[24]:


# preparation of the data

X_train, y_train = database.loc[:,'X1':'X5'], database['H']
ohe = OneHotEncoder(sparse=False)

X_ohe_train = ohe.fit_transform(X_train)

zero_data = np.zeros(shape=X_train.shape)
Radius_data = pd.DataFrame(zero_data, columns=['R1', 'R2', 'R3', 'R4', 'R5'])
Radius_data = X_train.applymap(return_radius)
Electron_data = pd.DataFrame(zero_data, columns=['E1', 'E2', 'E3', 'E4', 'E5'])
Electron_data = X_train.applymap(return_valen_el)
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(Radius_data) 
Radius_data.loc[:,:] = scaled_values
scaled_values = scaler.fit_transform(Electron_data) 
Electron_data.loc[:,:] = scaled_values
X_train_new=np.append(X_ohe_train, Radius_data, axis=1)
X_train_new=np.append(X_train_new, Electron_data, axis=1)


# In[25]:


# Supervised learning

model.fit(X_train_new,y_train)

# simple prediction
R2_train = model.score(X_train_new,y_train)
print("R^2 on training data =", R2_train)

y_train_pred = model.predict(X_train_new)

MAE_train = mean_absolute_error(y_train, y_train_pred)
print("MAE on test data = ", MAE_train)
MSE_train = mean_squared_error(y_train, y_train_pred)
print("MSE on test data = ", MSE_train)
RMSE_train = math.sqrt(MSE_train)
print("RMSE on test data = ", RMSE_train)


# In[26]:


x_tmp = np.linspace(min(min(y_train_pred), min(y_train)),max(max(y_train_pred),max(y_train)),100)
y_tmp = np.linspace(min(min(y_train_pred), min(y_train)),max(max(y_train_pred),max(y_train)),100)


plt.scatter(y_train_pred, y_train,c='g',label='$R^2$ = '+ 
            str(round(R2_train,2)) + ',\n MSE ='+ 
            str(round(MSE_train,2)))
plt.xlabel('Predicted (test data)')
plt.ylabel('Observed (test data)')
plt.plot(x_tmp,y_tmp,c='k')

#plt.title('neural network + Atomic Radius + Valence Electron')
plt.legend(loc=4)


# In[27]:


system_to_predict = ["Cr","Fe"]
#system_to_predict = ["Al","Co","Cr","Fe","Mn","Mo","Nb","Ni","Pt","Re","Ru","V","W","Zr"]

#function of prediction of a dataframe
def return_prediction(row):
    config=pd.DataFrame([[row['X1'],row['X2'],row['X3'],row['X4'],row['X5']]])
    temp = np.append(ohe.transform(config), config.applymap(return_radius_normed), axis=1) 
    prediction = np.append(temp, config.applymap(return_valen_el_normed), axis=1)
    return model.predict(prediction)[0]


# In[28]:


# preparation of the dataframe of New Compounds (NC)

NC = pd.DataFrame(columns=['num','X1', 'X2', 'X3', 'X4', 'X5', 'H_meV', 'H_kJ', 'a', 'c', 'ratio', 'vol', 
              'x_4f', 'x_8i1', 'y_8i1', 'x_8i2', 'y_8i2', 'x_8j', 'z_8j', 'mag'])

for A in system_to_predict:
    for B in system_to_predict:
        for C in system_to_predict:
            for D in system_to_predict:
                for E in system_to_predict:
                    #print(A,B,C,D,E)
                    NC = NC.append({'X1': A, 'X2': B, 'X3': C, 'X4': D, 'X5': E}, ignore_index=True)

NC['level'] = NC.apply(level_sys, axis=1)
NC.shape


# In[29]:


#estimation of the heat of formation

NC[NC.columns[7]] = NC.apply(return_prediction, axis=1)
NC.head()  


# In[30]:


# prediction of other data based on same model

colonne = [8,9] + list(range(12,19))
for col_num in colonne:
    col = database[database.columns[col_num]]
    model.fit(X_train_new,col)
    NC[NC.columns[col_num]] = NC.apply(return_prediction, axis=1)


# In[31]:


NC['H_meV']= NC['H_kJ']*9.6486
NC['ratio']= NC['c']/NC['a']
NC['vol']= NC['c']*NC['a']**2
NC['mag']= 0.0
NC.head() 


# ## (7) Writting files

# In[32]:


# database
import os, shutil
NC.to_csv('sigma-CrFe.zip', compression='zip')  


# In[33]:


# TDB
f = open('sigma-CrFe.TDB', 'w+')
for index, row in NC.iterrows():
    f.write(f" PARAM G(SIGMA,{row['X1']:2}:{row['X2']:2}:{row['X3']:2}:{row['X4']:2}:{row['X5']:2};0) 298 {row['H_kJ']*30000:.2f} ; 6000 N MLJCC !\n")
f.close()


# In[34]:


# VASP file

for index, row in NC.iterrows():
    path='predictions-VASP/'+row['X1']+row['X2']+row['X3']+row['X4']+row['X5']
    if not os.path.isdir(path):
        os.makedirs(path)
    #    
    # info
    f = open(path+'/info', 'w+')
    f.write(f"sigma - {index} - {row['X1']:3} {row['X2']:3} {row['X3']:3} {row['X4']:3} {row['X5']:3}\n")
    f.write(f"prediction:\n")
    #f.write(f"  F = {row['F']:8f}\n")
    #f.write(f"  H = {row['F']:8f}\n")
    f.write(f"  a = {row['a']:4f}\n")
    f.write(f"  c = {row['c']:4f}\n")
    f.close()
    #
    #
    # POSCAR
    f = open(path+'/POSCAR', 'w+')
    f.write(f"sigma - {index} - {row['X1']:3} {row['X2']:3} {row['X3']:3} {row['X4']:3} {row['X5']:3}\n")
    f.write(f"   1.0000000000000\n")
    f.write(f"     {row['a']:8f} 0.000000 0.000000\n")
    f.write(f"     0.000000 {row['a']:8f} 0.000000\n")
    f.write(f"     0.000000 0.000000 {row['c']:8f}\n")
    f.write(f"   2  4  8  8  8\n")
    f.write(f"Selective dynamics\n")
    f.write(f"Direct\n")  
    f.write(f"  0.000000 0.000000 0.000000 F F F   ! (2a)  CN12 - {row['X1']:3}\n")
    f.write(f"  0.500000 0.500000 0.500000 F F F   ! \n")                        
    f.write(f"  {row['x_4f']:8f} {row['x_4f']:8f} 0.000000 T T F   ! (4f)  CN15 - {row['X2']:3}\n")            
    f.write(f"  {1-row['x_4f']:8f} {1-row['x_4f']:8f} 0.000000 T T F   !\n")
    f.write(f"  {0.5-row['x_4f']:8f} {0.5+row['x_4f']:8f} 0.500000 T T F   !\n")
    f.write(f"  {0.5+row['x_4f']:8f} {0.5-row['x_4f']:8f} 0.500000 T T F   !\n")
    f.write(f"  {row['x_8i1']:8f} {row['y_8i1']:8f} 0.000000 T T F   ! (8i1) CN14 - {row['X3']:3}\n")
    f.write(f"  {1-row['x_8i1']:8f} {1-row['y_8i1']:8f} 0.000000 T T F   ! \n")
    f.write(f"  {0.5-row['y_8i1']:8f} {0.5+row['x_8i1']:8f} 0.500000 T T F   ! \n")
    f.write(f"  {0.5+row['y_8i1']:8f} {0.5-row['x_8i1']:8f} 0.500000 T T F   ! \n")
    f.write(f"  {0.5-row['x_8i1']:8f} {0.5+row['y_8i1']:8f} 0.500000 T T F   ! \n")
    f.write(f"  {0.5+row['x_8i1']:8f} {0.5-row['y_8i1']:8f} 0.500000 T T F   ! \n")
    f.write(f"  {row['y_8i1']:8f} {row['x_8i1']:8f} 0.000000 T T F   ! \n")
    f.write(f"  {1-row['y_8i1']:8f} {1-row['x_8i1']:8f} 0.000000 T T F   ! \n")
    f.write(f"  {row['x_8i2']:8f} {row['y_8i2']:8f} 0.000000 T T F   ! (8i2) CN12 - {row['X4']:3}\n")
    f.write(f"  {1-row['x_8i2']:8f} {1-row['y_8i2']:8f} 0.000000 T T F   ! \n")
    f.write(f"  {0.5-row['y_8i2']:8f} {-0.5+row['x_8i2']:8f} 0.500000 T T F   ! \n")
    f.write(f"  {0.5+row['y_8i2']:8f} {1.5-row['x_8i2']:8f} 0.500000 T T F   ! \n")
    f.write(f"  {1.5-row['x_8i2']:8f} {0.5+row['y_8i2']:8f} 0.500000 T T F   ! \n")
    f.write(f"  {-0.5+row['x_8i2']:8f} {0.5-row['y_8i2']:8f} 0.500000 T T F   ! \n")
    f.write(f"  {row['y_8i2']:8f} {row['x_8i2']:8f} 0.000000 T T F   ! \n")
    f.write(f"  {1-row['y_8i2']:8f} {1-row['x_8i2']:8f} 0.000000 T T F   ! \n")
    f.write(f"  {row['x_8j']:8f} {row['x_8j']:8f} {row['z_8j']:8f} T T T   ! (8j) CN14 - {row['X5']:3}\n")
    f.write(f"  {1-row['x_8j']:8f} {1-row['x_8j']:8f} {row['z_8j']:8f} T T T   ! \n")
    f.write(f"  {0.5-row['x_8j']:8f} {0.5+row['x_8j']:8f} {0.5+row['z_8j']:8f} T T T   ! \n")
    f.write(f"  {0.5+row['x_8j']:8f} {0.5-row['x_8j']:8f} {0.5+row['z_8j']:8f} T T T   ! \n")
    f.write(f"  {0.5-row['x_8j']:8f} {0.5+row['x_8j']:8f} {0.5-row['z_8j']:8f} T T T   ! \n")
    f.write(f"  {0.5+row['x_8j']:8f} {0.5-row['x_8j']:8f} {0.5-row['z_8j']:8f} T T T   ! \n")
    f.write(f"  {row['x_8j']:8f} {row['x_8j']:8f} {1-row['z_8j']:8f} T T T   ! \n")
    f.write(f"  {1-row['x_8j']:8f} {1-row['x_8j']:8f} {1-row['z_8j']:8f} T T T   ! \n")
    f.close()
    # POTCAR 
    pot_path='potpaw_PBE.54/'     
    filenames = [pot_path+row['X1']+'/POTCAR',pot_path+row['X2']+'/POTCAR',pot_path+row['X3']+'/POTCAR',pot_path+row['X4']+'/POTCAR',pot_path+row['X5']+'/POTCAR'] 
    with open(path+'/POTCAR', 'w') as outfile:
            for names in filenames:
                with open(names) as infile: 
                    outfile.write(infile.read()) 
                #outfile.write("\n") 


# In[ ]:





# In[ ]:




