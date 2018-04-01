# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:30:23 2018

@author: Student
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


train_data = pd.read_csv("C:\\Users\\Student\\Desktop\\applied\\train_data.csv")
test_data  = pd.read_csv("C:\\Users\\Student\\Desktop\\applied\\test_data.csv")

label_Y = np.array(train_data['y'])

#classification based on termfrequency matrix

New_f3 = [train_data['f3'],test_data['f3']];
Final_doc = pd.concat(New_f3)
del New_f3
TF_IDF = TfidfVectorizer()
feature_matrix = TF_IDF.fit_transform(Final_doc)
del Final_doc

TF_idf_train = feature_matrix[0:2656,:]
TF_idf_test = feature_matrix[2656:3321,:]



#sorting of cloumn based on their means
sorted_tfidf_index = np.array(TF_idf_train.mean(0)).reshape(153070,)
sorted_tfidf_index = sorted_tfidf_index.argsort()
new_tf_idf = TF_idf_train[:,sorted_tfidf_index[-10000:-1]]

sorted_tfidf_index_test = np.array(TF_idf_test.mean(0)).reshape(153070,)
sorted_tfidf_index_test = sorted_tfidf_index_test.argsort()
#taking first 10000 features
new_tf_idf_test = TF_idf_test[:,sorted_tfidf_index_test[-10000:-1]]

clf = LogisticRegression(solver='lbfgs',C=10)

clf.fit(new_tf_idf,label_Y)



y_proba = clf.predict_proba(new_tf_idf_test )

y_df = pd.DataFrame(y_proba) #label_Y_test is an array
y_df.index += 1
y_df.to_csv('mlclassificationresultsprediction2.csv')

