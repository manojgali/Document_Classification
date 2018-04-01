import pandas as pd 
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

import numpy as np
from sklearn import svm
from sklearn.decomposition import SparsePCA



    

    
def Supportvector(u_train, u_test, y_train,y_test):
    clf = svm.SVC(decision_function_shape='ovo',probability=True,C=20.0 )
    clf=clf.fit(u_train, y_train)
    arry = clf.predict_proba(u_test)
    k=sklearn.metrics.log_loss(y_test,arry)
    print("support vector machine logloss for 500 is :" ,k )
    
    #y_df = pd.DataFrame(arry) #label_Y_test is an array
    #y_df.index += 1
    #y_df.to_csv('mlclassificationresults.csv') 
    return 0;



#change your data set path as per your system.
train = pd.read_csv("C:\\Users\\Student\\Desktop\\applied\\train_data.csv")
test = pd.read_csv("C:\\Users\\Student\\Desktop\\applied\\test_data.csv")
#print(train.head())
#word=train[['f3']]

#the train data set:This dataset contains three features f1, f2 and f3. f1 and f2
# are categorical features. f3 contains a list of words represented as w1, w2, w3 etc.
# in a space separated format. Each w_i represents a word in english languages, 
#We have anonymized the data in each of our features f1, f2 and f3 to preserve 
#the privacy of the individuals whose data we are employing here.This dataset also
# contains a field called ‘y’ which is the class label.
#tokenizer term frequency id matrix


#the test data set:This dataset contains the same set of features as train dataset: f1, f2 and f3. 
#This dataset does not contain the class labels “y”. 
#You are to predict the class labels for this dataset .

###############term frequency documnent for cloumn 3 in train and test data set####

New_f3 = [train['f3'],test['f3']];
Final_doc = pd.concat(New_f3)
del New_f3
TF_IDF = TfidfVectorizer()
feature_matrix = TF_IDF.fit_transform(Final_doc)
del Final_doc

TF_idf_train = feature_matrix[0:2656,:].toarray()
TF_idf_test = feature_matrix[2656:3321,:]


#singular value decomposition 299
#here 299 is the new dimension for columns based on the formula mentined in column.py
svd = TruncatedSVD(n_components=299)



modifie=svd.fit_transform(TF_idf_train)


#FOR F1 column:
# the data contains in the form of 3 digit numbers as maximum so if there is any two digit 
#number or single digit number this conversion makes into three digit number.
array_f1 = np.zeros([2656,3])
for  i in range(0,len(train)):
     list1 = list(train['f1'][i])
     del list1[0]
     length = len(list1)
     list1 = ['0']*(3-length)+list1
     for j in range(0,3):
         array_f1[i,j] = int(list1[j])

#for f2 similarly it makes every data as four digit number.
array_f2 = np.zeros([2656,4])
for  i in range(0,len(train)):
     list1 = list(train['f2'][i])
     del list1[0]
     length = len(list1)
     list1 = ['0']*(4-length)+list1
     for j in range(0,4):
         array_f2[i,j] = int(list1[j])

#for test set
arrayf1_test = np.zeros([665,3])
for  i in range(0,len(test)):
     list1 = list(test['f1'][i])
     del list1[0]
     length = len(list1)
     list1 = ['0']*(3-length)+list1
     for j in range(0,3):
         arrayf1_test[i,j] = int(list1[j])

#for f2
arrayf2_test= np.zeros([665,4])
for  i in range(0,len(test)):
     list1 = list(train['f2'][i])
     del list1[0]
     length = len(list1)
     list1 = ['0']*(4-length)+list1
     for j in range(0,4):
         arrayf2_test[i,j] = int(list1[j])



p1=np.append(array_f2,modifie,axis=1)
u=np.append(array_f1,p1,axis=1)


#modifie_test=svd.fit_transform(x_test)
#p1_test=np.append(arrayf2_test,modifie_test,axis=1)
#u_test=np.append(arrayf1_test,p1_test,axis=1)


y=np.array(train['y'])

# train data cross validation data and test data
u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.15)
    

Supportvector(u_train, u_test, y_train,y_test)









   


