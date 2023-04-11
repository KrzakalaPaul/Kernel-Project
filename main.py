from kernel_class import Kernel
import numpy as np

## ------------------------------------ LOAD GRAM MATRICES ------------------------------------ # 
  
depth_wl=[1,2,3,4]
max_path_lenght=3

train_train_dic={}
validation_train_dic={}

for k in depth_wl:

    save_name=f'train_train_{k}'
    loaded=np.load('ComputeKernels\\gram_saves\\'+save_name+'.npz')

    for n in range(max_path_lenght+1):
        train_train_dic[(k,n)]=loaded[str(n)].astype(np.float64)    

    save_name=f'validation_train_{k}'
    loaded=np.load('ComputeKernels\\gram_saves\\'+save_name+'.npz')

    for n in range(max_path_lenght+1):
        validation_train_dic[(k,n)]=loaded[str(n)].astype(np.float64)

## ------------------------------------ KERNEL ------------------------------------ # 

class Combinaison_of_Kernel(Kernel):

    def __init__(self,alpha_k=1,alpha_n=1):

        self.kernel_train_train=np.zeros((6000,6000))
        self.kernel_validation_train=np.zeros((2000,6000))

        k_factors=alpha_k**np.array(depth_wl)
        k_factors=k_factors/np.sum(k_factors)
        n_factors=alpha_n**np.array(range(max_path_lenght+1))
        n_factors=n_factors/np.sum(n_factors)

        for k,c_k in zip(depth_wl,k_factors):
            for n,c_n in zip(range(max_path_lenght+1),n_factors):
                self.kernel_train_train+=c_k*c_n*(train_train_dic[(k,n)])
                self.kernel_validation_train+=c_k*c_n*(validation_train_dic[(k,n)])


## ------------------------------------ FineTuning ------------------------------------ # 

from finetuning import eval_model
'''
for alpha_k in [1]:
    for alpha_n in [0.1]:
        for C in [1e-2,1e-1,1]:
            for class_weight in [None,"balanced"]:   
                model=Combinaison_of_Kernel(alpha_k=alpha_k,alpha_n=alpha_n)
                print(f'params: {alpha_k},{alpha_n},{C},{class_weight}    score: {eval_model(model,C=C,class_weight=class_weight)}')

'''
## ------------------------------------ Predictions ------------------------------------ # 

import pandas as pd

alpha_k=1
alpha_n=0.1
class_weight="balanced"
C=1e-1

model=Combinaison_of_Kernel(alpha_k=alpha_k,alpha_n=alpha_n)
model.fit(train_indices=None,C=C,class_weight=class_weight)

validation_preds=model.pred(test_indices=None)

dataframe = pd.DataFrame({'Predicted' : validation_preds} )
dataframe.index += 1
dataframe.to_csv('validation_pred.csv',index_label='Id')
