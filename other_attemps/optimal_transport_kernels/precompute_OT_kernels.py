from utils.gram_precomputer import GramPrecomputer
from utils.graph import Graph_nx
import pickle as pkl
from .ot_distances import *

### ------------------- Load Data ------------------- ###

with open('KernelGraphChallenge\\save\\training_data.pkl', 'rb') as file:
    train_graphs = pkl.load(file)
with open('KernelGraphChallenge\\save\\test_data.pkl', 'rb') as file:
    test_graphs = pkl.load(file)
    
train_graphs=[Graph_nx(G,debug=True).to_array(50,3) for G in  train_graphs]
validation_graphs=[Graph_nx(G,debug=True).to_array(50,3) for G in test_graphs]

### ------------------- fGW distance ------------------- ###


alpha=0.5
n_iter=30

def distance_metric(G1,G2):
    return compute_mfGW(G1,G2,alpha=alpha,numItermax=n_iter)

def preprocessing(G):
    return G

Precomputer=GramPrecomputer(distance_metric,preprocessing,n_matrices=1)

Precomputer.estimate_time(train_graphs)

if __name__ == "__main__":


    '''
    print('Wasserstein Distance (exact)')
    kernel=Kernel(compute_W_exact)
    t=time()
    mat=kernel.sym_matrix(sample0+sample1)
    print(f'Computation time: {time()-t}')
    print(f'Accuracy: {kernel_quality(mat)}')

    print('')
    print('Wasserstein Distance + reg')
    kernel=Kernel(compute_W,reg=1e-1,tol=1e-6)
    t=time()
    mat=kernel.sym_matrix(sample0+sample1)
    print(f'Computation time: {time()-t}')
    print(f'Accuracy: {kernel_quality(mat)}')



    print('')
    print('Gromov Wasserstein Distance')
    kernel=Kernel(compute_GW,reg=1e-1,tol=1e-6)
    t=time()
    mat=kernel.sym_matrix(sample0+sample1)
    print(f'Computation time: {time()-t}')
    print(f'Accuracy: {kernel_quality(mat)}')

    print('')
    print('fused Gromov Wasserstein Distance')
    kernel=Kernel(compute_fGW,alpha=0.5,armijo=True,edge_labels=0)
    t=time()
    mat=kernel.sym_matrix(sample0+sample1)
    print(f'Computation time: {time()-t}')
    print(f'Accuracy: {kernel_quality(mat)}')
    
    print('')
    print('multi fused Gromov Wasserstein Distance')
    kernel=Kernel(compute_mfGW,alpha=0.5,numItermax=30)
    t=time()
    mat=kernel.sym_matrix(sample0+sample1)
    print(f'Computation time: {time()-t}')
    print(f'Accuracy: {kernel_quality(mat)}')
    '''

 
