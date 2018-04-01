# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:54:08 2018

@author: Student
"""

from sklearn.model_selection import KFold


def Supportvector(u_train, u_test, Ya_train,Ya_test):
    clf = svm.SVC(decision_function_shape='ovo',probability=True,C=1000.0 )
    clf=clf.fit(u_train, Ya_train)
    n=clf.score(u_train,Ya_train)
    p=clf.score(u_test,Ya_test)

    arry = clf.predict_proba(u_test)
    
    k=sklearn.metrics.log_loss(Ya_test,arry)
    return k,arry,n,p;
    
    
    
def select_rows(a,b):
    index_1  = np.array(np.where(y == a))
    index_2  = np.array(np.where(y == b))
    index = np.append(index_1,index_2)
    index.sort()
    gk= featue_matrix_train_data[index,:]
    gh = y[index]
    return gk,gh
                
#making 9c2 classifiers to build a decision tree.                 
gk,gh=select_rows(1,4)

store_list=[]#logloss list
feture_list=[]#index list
train_score_list=[]#train score list
test_score_list=[]#test score list 
for i in range(1,607,20):
    feture_list.append(i)
    u=gk[:,0:i]
    kf = KFold(n_splits=2)
    kf.get_n_splits(u)
    for train_index, test_index in kf.split(u):
        u_train, u_test = u[train_index], u[test_index]
        Ya_train, Ya_test = gh[train_index], gh[test_index]
       
        
    
    logloss,probability,train_score,test_score=Supportvector(u_train, u_test, Ya_train,Ya_test)
    store_list.append(logloss)
    
    train_score_list.append(train_score)
    test_score_list.append(test_score)
    plt.plot(feture_list,test_score_list,'b')
    plt.plot(feture_list,train_score_list,'g')
    plt.plot(feture_list,store_list,'r')