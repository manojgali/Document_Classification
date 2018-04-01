# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:49:52 2018

@author: Student
"""

#output prediction
y_pred = np.zeros([664,])
for i in range(0,len(y_test)):
    List_indices = [1,2,3,4,5,6,7,8,9]
    
    while len(List_indices)>1 :
        a = List_indices[0]
        b = List_indices[1]
        p = ['c','l','f',str(a),str(b)]
        p = ''.join(p)
        out = eval(p + ".predict(x_test[i,0:total_dictionary[p][0]])")
        if out==a:
            List_indices.remove(b)
        else:
            List_indices.remove(a)
    y_pred[i] = List_indices.pop()