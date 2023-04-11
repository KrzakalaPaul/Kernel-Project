### ------------------- Load Data ------------------- ###

import pickle as pkl

with open('training_data.pkl', 'rb') as file:
    train_graphs = pkl.load(file)
with open('test_data.pkl', 'rb') as file:
    validation_graphs = pkl.load(file)

### ------------------- Compute Gram Matrices ------------------- ###

from gram import GramComputer
from kernels import n_path
from label_augmentation import wl

for k in [4,3,2,1]:
    #k=3 
    n=5

    def gram_metric(G1,G2):
        return n_path(G1,G2,n)

    def preprocessing(G):
        return wl(G,depth=k)


    Precomputer=GramComputer(gram_metric,preprocessing,n_matrices=n+1)

    Precomputer.estimate_time(train_graphs)

    Precomputer.sym_matrix(train_graphs,save_name=f'train_train_{k}_noedgelabel')
    Precomputer.matrix(validation_graphs,train_graphs,save_name=f'validation_train_{k}_noedgelabel')