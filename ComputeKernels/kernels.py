import networkx as nx
from scipy import sparse

def get_product_graph(graph1,graph2):

    G1=graph1
    G2=graph2
    G12 = nx.Graph()

    # OPTIMIZED VERSION :
    for node1,data1 in G1.nodes(data=True):
        for node2,data2 in G2.nodes(data=True):
            if data1==data2:
                node12=(node1,node2)
                label12=data1['labels']
                G12.add_node(node12,label=label12)

    row = []
    col = []
    data = []

    for i,node12_a in enumerate(G12.nodes()):
        for j,node12_b in enumerate(G12.nodes()):

            node1_a,node2_a = node12_a
            node1_b,node2_b = node12_b

            if G1.has_edge(node1_a,node1_b):
                if G2.has_edge(node2_a,node2_b):
                    '''
                    # WITH EDGE LABEL
                    label1=G1.edges[node1_a,node1_b]['labels']
                    label2=G2.edges[node2_a,node2_b]['labels']
                    if label1==label2:
                        #G12.add_edge(node12_a,node12_b,label=label1)
                        row.append(i)
                        col.append(j)
                        data.append(1)
                    '''
                    # WITHOUT EDGE LABEL
                    row.append(i)
                    col.append(j)
                    data.append(1)
    n=len(G12)
    return sparse.coo_matrix((data, (row, col)), shape=(n, n))

from time import time 

def n_path(G1,G2,n=1):

    adj_matrix_12=get_product_graph(G1,G2)
    
    size,size=adj_matrix_12.shape
    if size==0 or adj_matrix_12.sum()==0:
        return [size]+[0]*n
    
    mat=None

    x=[size]

    for _ in range(n):
        if mat==None:
            mat=adj_matrix_12.copy()
        else:
            mat=mat@adj_matrix_12
        x.append(mat.sum())

    return x
