"""----------------------Finding Dimension value-------------------------""" 
#Uncomment this only when you need to find the Value of 'K'  
#  Optimal Value of K found was '318'                    
"""1) Now the features of feature_matrix are to be reduced to certain value 'K'. 
   2) This is done by eigen vector decomposition  of C*CT where C is feature 
       matrix and CT is its transpose."""
"""    


"""""
####refer LSA(Latent Semantic Analysis)  better understanding)

import scipy
scipy.sparse.csr_matrix.transpose(feature_matrix)
CCT = feature_matrix*scipy.sparse.csr_matrix.transpose(feature_matrix)
Sum_CCT = 0
for i in range(0,CCT.shape[0]):
    Sum_CCT = Sum_CCT + CCT[i,i]
from  scipy.sparse.linalg import eigs
vals, vecs = eigs(CCT, k=1000)
Sum_VECS = 0
for K in range(0,1000):
    Sum_VECS = Sum_VECS + abs(vals[K])
    if Sum_VECS >= 0.8*Sum_CCT:
        break;



