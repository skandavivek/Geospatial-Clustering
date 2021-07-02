from matplotlib import pyplot as plt
import osmnx as ox
import numpy as np
import pandas as pd

def plot1(G,ids,labels,name):

    nc = ["w" for node in G.nodes()]
    ns = [0 for node in G.nodes()]
    nodes_G = list(G.nodes)
    #cols = ["C0", "C1", 'C2']
    ids=np.array(ids)
    ids=ids[labels>=0]
    labels=labels[labels>=0]
    la, co = np.unique(labels, return_counts=True)
    la_l=la[np.argsort(-co)]
    nodes_G=np.array(nodes_G)

    for c in range(0, len(ids)):
        # print(c)
        nc[np.where(nodes_G == ids[c])[0][0]] = 'C'+str(int(la_l[int(labels[c])]%7))
        ns[np.where(nodes_G == ids[c])[0][0]] = 2

    ox.plot_graph(G, edge_linewidth=0.1, node_size=ns, node_color=nc, bgcolor='w',save=True, filepath=name, dpi=600)

    return(nc,ns)


