# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:09:59 2018

@author: yang
"""
import numpy as np
import csv

def read_image(row):
    image = np.zeros((40,40))
    flag = 1
    for i in range(40):
        for j in range(40):
            image[i][j] = row[flag]
            flag+=1
    return image

def read_data(filepath):
    features = []
    labels = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            features.append(read_image(row))
            labels.append(row[-1])
    return np.array(features),np.array(labels)

def read_data_test(filepath):
    features = []
    ids = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            features.append(read_image(row))
            ids.append(row[0])
    return np.array(features), np.array(ids)
            
    

        


    