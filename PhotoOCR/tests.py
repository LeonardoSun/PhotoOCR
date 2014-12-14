# encoding:utf-8
'''
Created on 2014年12月14日

@author: leonardo
'''
import numpy as np
from Source.neurals.cortex import Cortex, NeuralRelation
from Source.neurals.indexable import NestingArrayIndexable

training_data_dic={}
training_data_shape = (32, 32)
def load_data():
    file_name_template = '/media/leonardo/unknow/AI/Machine Learning in Action/machinelearninginaction/Ch02/trainingDigits/%s_0.txt'
    for num in range(0,10):
        f= open(file_name_template % num, 'r')
        lines = f.readlines()
        f.close()
        d = []
        for l in lines:
            slst = []
            for s in l:
                if s.strip():
                    slst.append(int(s.strip()))
            d.append(slst)
        training_data_dic[num] = d

load_data()

level_1_cortex = Cortex(NestingArrayIndexable(training_data_dic[0], training_data_shape))

level_2_data = []
for y in range(level_1_cortex.shape):
    row = []
    for x in range(level_1_cortex.shape):
        row.append(NeuralRelation(level_1_cortex, x, y))
    level_2_data.append(row)
    
level_2_cortex = Cortex(NestingArrayIndexable(level_2_data, level_1_cortex.shape))
