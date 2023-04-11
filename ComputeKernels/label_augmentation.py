import networkx as nx
import numpy as np




def wl(G: nx.classes.graph.Graph, depth: int): 

    if depth==0:
        return G
    else:
        NewG=nx.Graph()

        for node,data in G.nodes(data=True):
            
            old_labels=data['labels']
            label=[]

            for neighbor,edge_attributes in G[node].items():

                edge_label=str(edge_attributes['labels'][0])
                neighbor_label=str(G.nodes[neighbor]['labels'])
                label.append(edge_label+neighbor_label)
            label.sort()

            NewG.add_node(node, labels=old_labels+[hash('-'.join(label))])

        NewG.add_edges_from(G.edges(data=True))

        return wl(NewG, depth=depth-1)


if __name__=='__main__':

    G=nx.Graph()

    G.add_node(0,labels=[1])
    G.add_node(1,labels=[1])
    G.add_node(2,labels=[1])
    G.add_node(3,labels=[2])
    G.add_node(4,labels=[1])

    G.add_edge(0,1,labels=[1])
    G.add_edge(1,2,labels=[1])
    G.add_edge(2,3,labels=[1])
    G.add_edge(3,4,labels=[1])

    G=wl(G,depth=3)

    for node,data in G.nodes(data=True):
        print(data['labels'])


