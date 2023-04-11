import numpy as np
from time import time

def save(matrix_list,save_name,dtype=np.float64):
    dic={}
    for k,matrix in enumerate(matrix_list):
        dic[str(k)]=matrix.astype(dtype)
    np.savez_compressed('gram_saves\\'+save_name,**dic)

def load(save_name):
    loaded=np.load('gram_saves\\'+save_name+'.npz')
    matrix_list=[]
    for k in loaded.keys():
        matrix_list.append(loaded[k].astype(np.float64))
    return matrix_list


class GramComputer():

    def __init__(self,gram_metrics,label_augment,n_matrices=1):
        
        self.preprocessing=label_augment
        self.metrics=gram_metrics
        self.n_matrices=n_matrices
    

    def estimate_time(self,list_graph):
         
        sample=50
        N=len(list_graph)
        list_graph=list_graph[:sample]

        t1=time()
        list_graph=[self.preprocessing(graph) for graph in list_graph]
        t1=time()-t1

        t2=time()
        for i,graph1 in enumerate(list_graph[:sample]):
            for k,graph2 in enumerate(list_graph[:sample][i+1:]):
                self.metrics(graph1,graph2)
        t2=time()-t2

        t1=t1*(N/sample)
        t2=t2*(N/sample)**2
        t=t1+t2

        print( f'label augmentation = {t1/(60*60)} hours')
        print( f'Matrix = {t2/(60*60)} hours')
        print( f'Total = {t/(60*60)} hours')


    def sym_matrix(self,list_graph,save_name=None,save_period_min=10):

        list_graph=[self.preprocessing(graph) for graph in list_graph]
           
        N=len(list_graph)

        matrices=[np.zeros((N,N)) for _ in range(self.n_matrices)]

        t0=time()

        for i,graph1 in enumerate(list_graph):
            for k,graph2 in enumerate(list_graph[i:]):
                j=k+i

                all_x=self.metrics(graph1,graph2)
                
                for matrix,x in zip(matrices,all_x):
                    matrix[i,j]=x
                    matrix[j,i]=x

                if save_name!=None:
                    if time()-t0>save_period_min*60:
                        save(matrices,save_name)
                        t0=time()

        if save_name!=None:
            save(matrices,save_name)
            
        return matrices


    def matrix(self,list_graph1,list_graph2,save_name=None,save_period_min=10):

        list_graph1=[self.preprocessing(graph) for graph in list_graph1]
        list_graph2=[self.preprocessing(graph) for graph in list_graph2]

        N1=len(list_graph1)
        N2=len(list_graph2)

        matrices=[np.zeros((N1,N2)) for _ in range(self.n_matrices)]

        t0=time()

        for i,graph1 in enumerate(list_graph1):
            for j,graph2 in enumerate(list_graph2):

                all_x=self.metrics(graph1,graph2)

                for matrix,x in zip(matrices,all_x):
                    matrix[i,j]=x

                if save_name!=None:
                    if time()-t0>save_period_min*60:
                        save(matrices,save_name)
                        t0=time()

        if save_name!=None:
            save(matrices,save_name)

            
        return matrices