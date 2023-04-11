from ot.gromov import fused_gromov_wasserstein2,entropic_gromov_wasserstein2
from .mfGW import mixed_fused_gromov_wasserstein2
from ot.bregman import sinkhorn2 
from ot.lp import emd2
from copy import copy
import numpy as np
from utils.graph import Graph_array,Graph_nx


def compute_W(graph1:Graph_array,graph2:Graph_array, reg=1e-1,**kwargs):

    n_1=graph1.len()
    p_1=np.full(n_1, 1/n_1)

    n_2=graph2.len()
    p_2=np.full(n_2, 1/n_2) 

    inter_cost=graph1.X@graph2.X.T

    return sinkhorn2(p_1, p_2, inter_cost, reg=reg, **kwargs )



def compute_W_exact(graph1:Graph_array,graph2:Graph_array):

    n_1=graph1.len()
    p_1=np.full(n_1, 1/n_1)

    n_2=graph2.len()
    p_2=np.full(n_2, 1/n_2) 

    inter_cost=graph1.X@graph2.X.T

    return emd2(p_1, p_2, inter_cost)


def compute_GW(graph1:Graph_array,graph2:Graph_array, reg=1e-1, edge_labels='all',**kwargs):

    n1=graph1.len()
    p1=np.full(n1, 1/n1)

    if edge_labels=='all':
        C1=sum(graph1.Cs)
    else:
        C1=C1[edge_labels]

    n2=graph2.len()
    p2=np.full(n2, 1/n2) 

    if edge_labels=='all':
        C2=sum(graph2.Cs)
    else:
        C2=C2[edge_labels]

    return entropic_gromov_wasserstein2(C1.toarray(), C2.toarray(), p1, p2, 'square_loss', reg, **kwargs)

def compute_fGW(graph1:Graph_array,graph2:Graph_array, alpha=0.5, edge_labels='all',**kwargs):

    n1=graph1.len()
    p1=np.full(n1, 1/n1)

    if edge_labels=='all':
        C1=sum(graph1.Cs)
    else:
        C1=graph1.Cs[edge_labels]

    n2=graph2.len()
    p2=np.full(n2, 1/n2) 

    if edge_labels=='all':
        C2=sum(graph2.Cs)
    else:
        C2=graph2.Cs[edge_labels]

    inter_cost=graph1.X@graph2.X.T

    return fused_gromov_wasserstein2(inter_cost,C1.toarray(), C2.toarray(), p1, p2, loss_fun= 'square_loss', alpha=alpha, **kwargs)

def compute_mfGW(graph1:Graph_array,graph2:Graph_array, alpha=0.5,**kwargs):

    n1=graph1.len()
    p1=np.full(n1, 1/n1)
    Cs1=[C1.toarray() for C1 in graph1.Cs ]

    n2=graph2.len()
    p2=np.full(n2, 1/n2) 
    Cs2=[C2.toarray() for C2 in graph2.Cs ]

    inter_cost=graph1.X@graph2.X.T

    return mixed_fused_gromov_wasserstein2(inter_cost,Cs1, Cs2, p1, p2, loss_fun= 'square_loss', alpha=alpha, **kwargs)
