import numpy as np


def vector_to_str(vector):
    words=[]
    for word in vector:
        words.append(str(word))
    words_str=" ".join(words)
    return words_str

def show(clusters,sentences_map):
    cluster_num=1
    with open('result.txt','w+',encoding='utf-8') as fp:
        for cluster in clusters:
            fp.write('\n--------------------------------------------\n')
            fp.write('cluster '+str(cluster_num)+'\n')
            for vector in cluster:
                fp.write(sentences_map[vector_to_str(vector)]+'\n')
            cluster_num+=1