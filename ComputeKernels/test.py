import matplotlib.pyplot as plt 
import numpy as np

'''
plt.figure()
X=np.array(range(6))

for alpha in np.linspace(0.5,1.5,20):
    Y=alpha**X
    plt.plot(X,Y/np.sum(Y),label=f'{alpha}')
plt.show()
'''
'''
from scipy import sparse


row = np.array([0, 3, 1, 0])
col = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
M = sparse.coo_matrix((data, (row, col)), shape=(5, 5))

print(M.toarray())

print(M.shape)
'''

from networkx import Graph

G1=Graph()
G1.add_node(1,labels=[1])
G1.add_node(2,labels=[2])
G1.add_node(3,labels=[2])
G1.add_node(4,labels=[1])
G1.add_node(5,labels=[1])

G1.add_edge(1,2,labels=['a'])
G1.add_edge(2,3,labels=['a'])
G1.add_edge(3,4,labels=['a'])
G1.add_edge(3,5,labels=['b'])


G2=Graph()
G2.add_node(1,labels=[1])
G2.add_node(2,labels=[2])
G2.add_edge(1,2,labels=['a'])

from kernels import get_product_graph
from label_augmentation import wl

G1=wl(G1,depth=0)
G2=wl(G2,depth=0)


adj_matrix=get_product_graph(G1,G2)
print(adj_matrix.toarray())

from kernels import n_path
gram=n_path(G1,G2,n=4)
print(gram)