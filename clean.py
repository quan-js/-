__author__ = 'ccqy'
# -*- coding:utf-8 -*-

import os
import re
import numpy as np
import pandas as pd

def clean_train(train_data_path='/home/s-19/Test/train',out_train_filename='train_cleaned'):
        train_filenames=os.listdir(train_data_path)
        train_data_list=[]
	for train_filename in train_filenames:
	    if not train_filename.endswith('.txt'):
		    continue
	    train_file=os.path.join(train_data_path,train_filename)
	    lable=int(train_filename[0])
	    with open(train_file,'r') as f :
		   lines=f.read().splitlines()
	    lables=[lable]*len(lines)
	    labels_series=pd.Series(lables)
            lines_series=pd.Series(lines)
	    data_pd=pd.concat([labels_series,lines_series],axis=1)
            train_data_list.append(data_pd)
	train_pd=pd.concat(train_data_list,axis=0)
	train_pd.columns=['lable','text']
	train_pd.to_csv(os.path.join(train_data_path,out_train_filename),index=None,encoding='utf-8',header=True)

def clean_test(test_data_path='/home/s-19/Test/test',test_filename='test1.csv',out_test_filename='test_cleaned'):
       test_file=os.path.join(test_data_path,test_filename)
       with open(test_file,'r') as f :
           lines=f.read().splitlines()
       lines_series=pd.Series(lines)
       test_data_list=pd.Series(lines_series,name='text')
       test_data_list.to_csv(os.path.join(test_data_path,out_test_filename),index=None,encoding='utf-8',header=True)

if  __name__=="__main__":
    clean_train()
    clean_test()
