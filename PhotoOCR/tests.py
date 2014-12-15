# encoding:utf-8
'''
Created on 2014年12月14日

@author: leonardo
'''
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

content = []
for y_index in level_2_cortex.shape[1]:
    for x_index in level_2_cortex.shape[0]:
            rel_neu = level_2_cortex.get(x_index, y_index)
            # define background
            if ((rel_neu.up is None or rel_neu.up == 0) and
                (rel_neu.down is None or rel_neu.down == 0) and
                (rel_neu.left is None or rel_neu.left == 0) and
                (rel_neu.right is None or rel_neu.right == 0) and
                (rel_neu.up_left is None or rel_neu.up_left == 0) and
                (rel_neu.down_left is None or rel_neu.down_left == 0) and
                (rel_neu.up_right is None or rel_neu.up_right == 0) and
                (rel_neu.down_right is None or rel_neu.down_right == 0)):
                # background
                pass
            else:
                # get content (wipe off background)
                content.append(x_index, y_index)

def Map(content):
    pass

# connect content (record related pattern)
level_3_cortex = Map(content)

# abstract connected pattern (include low level feature; count the appearance of these feature; etc.)

# loop training set, get multiple data.

# find difference

# further abstract

# basically, we have a point to represent pen ink.
# using machine to write number with this point.
# before writing, we learn the basic structure of numbers.
# and the character randomly and slightly differ from the structure.
# after writing a lot examples, we can write properly,
# and can recognize hand writing as well.