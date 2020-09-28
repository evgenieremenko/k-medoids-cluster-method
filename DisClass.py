#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:38:52 2020

@author: NoufAlghanmi
"""


#from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

def cdist_with_less_ram(X1, X2, category = "dice", width = None):
    """
    this function will find the similarity matrix using `cdist` of scipy but this will take less space the original one
    """
    size1, size2 = X1.shape[0], X2.shape[0]

    if width is None: width = min(size1, size2)//512
    elif width == 1: width = 2
    width = max(2, width)

    if X1.shape[1] != X2.shape[1]:
        raise ValueError("Given X1 and X2 should have same number of columns. i.e. X1.shape[1]!=X2.shape[1]")
    
    out = np.empty((size1, size2), np.float16)
    iterations1 = size1//width if size1%width==0 else size1//width+1
    iterations2 = size2//width if size2%width==0 else size2//width+1
    
    for i in range(iterations1):
        for j in range(iterations2):
            sub_a = X1[width*i:width*i+width]
            sub_b = X2[width*j:width*j+width]
            temp = cdist(sub_a,sub_b, category)
            out[width*i:width*i+width,width*j:width*j+width] = temp
    return out

class Distance_class(object):
    def __init__(self,X,X_categrical,X_numerical,w=[1,1]):
        self.w=w
        self.categrical = X_categrical
        self.numerical  = X_numerical
        self.X_categrical = X.loc[:, X_categrical]
        self.X_numerical  = X.loc[:, X_numerical]

    def fit(self):
        Category_distance  = cdist_with_less_ram(self.X_categrical.values, self.X_categrical.values, 'dice')
        Numerical_distance = cdist_with_less_ram(self.X_numerical.values, self.X_numerical.values, 'euclidean')
        FM=(Category_distance+Numerical_distance)/2
        return FM
    
    def compute_dist_from_medoids(self, X_test, medoids):
        Category_distance=cdist_with_less_ram(X_test.loc[:, self.categrical].values, self.X_categrical.loc[medoids, :],'dice')
        Numerical_distance=cdist_with_less_ram(X_test.loc[:, self.numerical].values, self.X_numerical.loc[medoids, :], 'euclidean')
        FM=(Category_distance+Numerical_distance)/2
        return pd.DataFrame(FM)
