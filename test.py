import torch
from torchviz import make_dot
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from ast import literal_eval
from xml_reader import XmlProcessor

xml_prefix_dict = {'fin': 'AS_TrainingSet_NLF_NewsEye_v2/',
                       'fre': 'AS_TrainingSet_BnF_NewsEye_v2/'}
with open('train_test_file_record/'+'train'+'_file_list_'+'fin'+'.txt', 'r') as file:
    file_list = literal_eval(file.readlines()[0])
file_list = [xml_prefix_dict['fin']+x for x in file_list]
print(file_list)
lenth = []
max = 0
for x in file_list:
    a = set([x['reading_order'] for x in XmlProcessor(1, x).get_annotation()])
    print(a)
    lenth.append(a.__len__())
    if a.__len__() > max:
        max = a.__len__()
print(max)
lenth.sort()
print(lenth)