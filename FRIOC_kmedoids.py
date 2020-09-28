#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:53:13 2020

@author: NoufAlghanmi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:49:34 2020

@author: NoufAlghanmi
"""

from pyclustering.cluster.kmedoids import kmedoids
import numpy as np
import random

class FRIOC_kmedoids:
    def __init__(self, X,Y,initial_medoids,threshold=1.5):
        self.X = X
        self.Y=Y
        self.threshold=threshold
        self.initial_medoids=initial_medoids

    def fit(self):
        Final_cluster=[]
        Temp_cluster=[]
        ToCheck_cluster=[]
        #threshold=0.9529#0.01 #0.5
        K=4#int(y.max()-y.min()/threshold)
        Final_medoids=[]
        Check_medoids=[]
        Temp_medoids=[]
        
        kmedoids_instance = kmedoids(self.X.values, self.initial_medoids,ccore=False, 
                                     data_type='distance_matrix')
        # run cluster analysis and obtain results
        kmedoids_instance.process()
        ToCheck_cluster = kmedoids_instance.get_clusters()
        Check_medoids=kmedoids_instance.get_medoids()
        OC=[]

        for i in range(len(Check_medoids)):
            STD=np.std(self.Y.iloc[ToCheck_cluster[i]]) #it is a number 
            if STD<=self.threshold:
                Final_cluster.append(ToCheck_cluster[i])
                Final_medoids.append(Check_medoids[i])
            else:
                Temp_cluster.append(ToCheck_cluster[i])
                Temp_medoids.append(Check_medoids[i])
        ToCheck_cluster=Temp_cluster
        Check_medoids=Temp_medoids
        
        while ToCheck_cluster:
            L=len(ToCheck_cluster)
            Temp_cluster=[]
            Temp_medoids=[]
            for i in range(0,L):                 
                 list= ToCheck_cluster[i]
                 if len(list)==0: continue
                 if len(list)<=2:
                     Final_cluster.append(list)
                     Final_medoids.append(Check_medoids[i])
                     continue

                 OC=self.Y.iloc[list]
                 STD=np.std(OC)
                 if STD<=self.threshold:
                     Final_cluster.append(list)
                     Final_medoids.append(Check_medoids[i])
                 else:
                     data=self.X.iloc[list, list]
                     new_medoids=random.sample(range(len(list)), K)                     
                     kmedoids_instance = kmedoids(data.values, new_medoids, ccore=False,
                                                  data_type='distance_matrix')
                     # run cluster analysis and obtain results
                     kmedoids_instance.process()                    
                     cluster=kmedoids_instance.get_clusters()
                    
                     for i in range(len(cluster)):
                         for j in range(len(cluster[i])):
                             cluster[i][j]=list[cluster[i][j]]
                         Temp_cluster.append(cluster[i])
                     
                     medoids=kmedoids_instance.get_medoids()
                     for i in range(len(medoids)):
                         Temp_medoids.append(list[medoids[i]])

            ToCheck_cluster=Temp_cluster
            Check_medoids=Temp_medoids


        return Final_medoids,Final_cluster
