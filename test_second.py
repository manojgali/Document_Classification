# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:11:53 2018

@author: Student


"""

import pandas as pd 
import sklearn

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import pickle




def select_rows(a,b):
    index_1  = np.array(np.where(y_train == a))
    index_2  = np.array(np.where(y_train == b))
    index = np.append(index_1,index_2)
    index.sort()
    Final_feature_matrix = x_train[index,:]
    Label_Y_Final = y_train[index]
    return Final_feature_matrix,Label_Y_Final

train = pd.read_csv("C:\\Users\\Student\\Desktop\\applied\\train_data.csv")
test = pd.read_csv("C:\\Users\\Student\\Desktop\\applied\\test_data.csv")



New_f3 = [train['f3'],test['f3']];
Final_doc = pd.concat(New_f3)
del New_f3
TF_IDF = TfidfVectorizer()
feature_matrix = TF_IDF.fit_transform(Final_doc)
del Final_doc



svd = TruncatedSVD(n_components=600) # K-value is used
feature_f3 = svd.fit_transform(feature_matrix)



# ####### arrays of F1 and F2 are made by splitting them into individual characters i.e 'a234' is made '2','3','4' 
array_f1 = np.zeros([2656,3])
for  i in range(0,len(train)):
     list1 = list(train['f1'][i])
     del list1[0]
     length = len(list1)
     list1 = ['0']*(3-length)+list1
     for j in range(0,3):
         array_f1[i,j] = int(list1[j])

array_f2 = np.zeros([2656,4])
for  i in range(0,len(train)):
     list1 = list(train['f2'][i])
     del list1[0]
     length = len(list1)
     list1 = ['0']*(4-length)+list1
     for j in range(0,4):
         array_f2[i,j] = int(list1[j])
         
array_f1_test = np.zeros([665,3])
for  i in range(0,len(test)):
     list1 = list(test['f1'][i])
     del list1[0]
     length = len(list1)
     list1 = ['0']*(3-length)+list1
     for j in range(0,3):
         array_f1_test[i,j] = int(list1[j])

array_f2_test = np.zeros([665,4])
for  i in range(0,len(test)):
     list1 = list(test['f2'][i])
     del list1[0]
     length = len(list1)
     list1 = ['0']*(4-length)+list1
     for j in range(0,4):
         array_f2_test[i,j] = int(list1[j])
         
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
array_f1 = Scaler.fit_transform(array_f1)  
array_f2 = Scaler.fit_transform(array_f2)
array_f1_test = Scaler.fit_transform(array_f1_test)
array_f2_test = Scaler.fit_transform(array_f2_test)     
del list1,i,j

# 2) Final_train_matrix 

feature_matrix_train = feature_f3[0:len(train),:]
feature_matrix_test = feature_f3[len(train):len(train)+len(test)]
feature_matrix_train = np.c_[array_f1,array_f2,feature_matrix_train]
feature_matrix_test = np.c_[array_f1_test,array_f2_test,feature_matrix_test]
#del array_f1,array_f2,array_f1_test,array_f2_test

#never forget to convert data frame into numpy  
y=np.array(train['y'])
####calculation of mutual information of every column w.r.t to y
k=sklearn.feature_selection.mutual_info_classif(feature_matrix_train, y)
#sorting in decreasing order
p=(-k).argsort()

featue_matrix_train_data=feature_matrix_train[:,p]

x_train, x_test, y_train, y_test = train_test_split(featue_matrix_train_data,y, test_size=0.25, random_state=0)


      

total_dictionary={}
total_dictionary['clf12'] = [140,0.9165,45]
total_dictionary['clf13'] = [50,0.9599,100]
total_dictionary['clf15'] = [170,0.8549,150]
total_dictionary['clf16'] = [210,0.8961,300]
total_dictionary['clf68'] = [100,0.9489,50]
total_dictionary['clf67'] = [190,0.9511,400]
total_dictionary['clf59'] = [115,0.9642,30]
total_dictionary['clf58'] = [20,0.9282,100]
total_dictionary['clf26'] = [110,0.9259,1000]
total_dictionary['clf36'] = [100,0.9553,300]
total_dictionary['clf45'] = [90,0.8963,1000]
total_dictionary['clf27'] = [140,0.79519145146927872,1000]
total_dictionary['clf14'] = [130,0.81455633100697911,1000]



with open('Dict_final_36.pkl', 'rb') as f:   
     dict = pickle.load(f)


total_dictionary.update(dict) 

###########################################pickle##########################
#import pickle# now you can save it to a filewith open('clf16.pkl', 'wb') as f:    pickle.dump(clf16, f)    
#with open('clf16.pkl', 'rb') as f:   
     #clf16 = pickle.load(f)
#with open('clf67.pkl', 'wb') as f:
    #pickle.dump(clf67, f) 
################################################################################# 


   
key = list(total_dictionary.keys())
for i in key:
    k = list(i)
    Final_feature_matrix,Label_Y_Final = select_rows(int(k[-1]),int(k[-2]))
    exec(i + " = sklearn.svm.SVC(C=total_dictionary[i][2], kernel='linear', probability=True).fit(Final_feature_matrix[:,0:total_dictionary[i][0]], Label_Y_Final)")
    del k



    

 
