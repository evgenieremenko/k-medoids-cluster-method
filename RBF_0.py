#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 00:22:24 2020

@author: NoufAlghanmi
"""

import pandas as pd
import numpy as np
from DisClass import Distance_class
from FRIOC_kmedoids import FRIOC_kmedoids
from numpy.random import randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def compute_cluster_stds(X,clusters,medoids):
    stds=[]
    new_medoids=[]
    for i in range(len(medoids)):
        if len(clusters[i])>=2:
            cluster=X.iloc[clusters[i]]
            cluster=cluster[medoids[i]]
            stds.append(np.sqrt(np.mean(cluster**2)))
            new_medoids.append(medoids[i])
    return stds,new_medoids

def normalizer(X_tr, X_te, numerical_columns):
    """
    X_tr: train dataset, if is a dataframe
    X_te: test dataset, if is a dataframe
    numerical_columns: list of columns that has numerical values
    """
    x_tr = X_tr.loc[:, numerical_columns]
    x_te = X_te.loc[:, numerical_columns]
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr.values)
    x_te = scaler.transform(x_te.values)
    X_tr.loc[:, numerical_columns] = x_tr
    X_te.loc[:, numerical_columns] = x_te
    return X_tr, X_te

def kernel_function_2( data_point,std):
    print("np.exp(-std*np.sum((data_point**2))):",np.exp(-std*(data_point**2)))
    return np.exp(-std*np.sum((data_point**2)))

def compute_interpolation_matrix(X,stds):
    return np.exp(-((X**2)/stds))

df = pd.read_csv("./autos_processed.csv",index_col=0)
y=df['price']
df.drop(['price'],axis=1,inplace=True)


categorical=[ 'vehicleType', 'gearbox', 'model', 'monthOfRegistration', 'fuelType', 'brand','notRepairedDamage']
numerical=['yearOfRegistration', 'powerPS','kilometer']

print("Autos normalized data set result threshole=0.5")
y=(y-y.min())/(y.max()-y.min())
MSE_train=[]
RMSE_train=[]
MSE_test=[]
RMSE_test=[]
no_centers=[]

for i in range(10):
    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(df):
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # let's reset the idices
        X_train.reset_index(drop = True, inplace= True);  X_test.reset_index(drop = True, inplace= True)
        y_train.reset_index(drop = True, inplace= True);  y_test.reset_index(drop = True, inplace= True)
        # nomalizing the numerical values
        X_train, X_test = normalizer(X_train, X_test, numerical)
        
        print("findind similarity matrix...",  end = " ")
        # Compute fused distance matrix and convert it to DataFrame
        Distance_class_instance=Distance_class(X_train,categorical,numerical)
        FM=Distance_class_instance.fit() #X_train.index.values.tolist()
        X_train=pd.DataFrame(FM)
        print("Done!")
        # RUn Kmedoids alghorithm
        threshold=0.5# y_train.min
        
        print("training....", end = " ")
        initial_medoids=randint(0, X_train.shape[0], 8).tolist()
        kmedoids_instance= FRIOC_kmedoids(X_train,y_train,initial_medoids,threshold=threshold)
        medoids,clusters=kmedoids_instance.fit()
        print("Done!")
        
        print("predicting and calcuating performace for train data...", end = " ")
        stds,medoids=compute_cluster_stds(X_train,clusters,medoids)
        #Compute inetrplation matrix for RBF
        G=compute_interpolation_matrix(X_train[medoids],stds)
        #w = np.dot(np.linalg.pinv(G), y_train.values)
        GTG= np.dot(G.T,G) 
        GTG_inv= np.linalg.inv(GTG.astype(np.float32)).astype(np.float16)
        fac= np.dot(GTG_inv,G.T)
        weights= np.dot(fac,y_train)
        y_pred=np.dot(G, weights)
        
        no_centers.append(len(medoids))
        MSE_train.append(mean_squared_error(y_train,y_pred))
        RMSE_train.append(mean_squared_error(y_train,y_pred,squared=False))
        print("Done!")
        
        print("predicting and calcuating performace for train data...", end = " ")
        FM_test=Distance_class_instance.compute_dist_from_medoids(X_test,medoids)
        G=compute_interpolation_matrix(FM_test,stds)
        prediction=np.dot(G, weights)
        MSE_test.append(mean_squared_error(y_test,prediction))
        RMSE_test.append(mean_squared_error(y_test,prediction,squared=False))
        print("Done!")


print("len(MSE_train):",len(MSE_train))
print("len(MSE_test):",len(MSE_test))


print("\nmean mse train error:{:.4f}".format(np.mean(MSE_train)))
print("std mse train error:{:.4f}".format(np.std(MSE_train)))


print("\nmean rmse train error:{:.4f}".format(np.mean(RMSE_train)))
print("std rmse train error:{:.4f}".format(np.std(RMSE_train)))


print("\nmean mse testing error:{:.4f}".format(np.mean(MSE_test)))
print("std mse testing error:{:.4f}".format(np.std(MSE_test)))


print("\nmean rmse testing error:{:.4f}".format(np.mean(RMSE_test)))
print("std rmse testing error:{:.4f}".format(np.std(RMSE_test)))

print("\nno of centers mean:{:.4f}".format(np.mean(no_centers)))
print("no of centers std:{:.4f}".format(np.std(no_centers)))

