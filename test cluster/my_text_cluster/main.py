import json
from preprocess import get_word_vectors
from preprocess import word_vectors_to_str
from cluster import KMeans
from show import show
import numpy as np

if __name__ == '__main__':
    data_jsons = []
    with open('dev.json', 'r') as fp:
        for jsonstr in fp.readlines():
            data_jsons.append(json.loads(jsonstr))
    # 取出sentence和labels部分，打包成列表
    sentences = [data_json['sentence'] for data_json in data_jsons]
    labels = [data_json['label'] for data_json in data_jsons]
    # 得到了每个句子中每个字的坐标
    word_vectors = get_word_vectors(sentences)
    #将句子的向量和句子本身进行配对
    sentences=[i+j for i,j in zip(labels,sentences)]
    #将word_vectors转化为字符串数组，数据量大了str会不好用，所以专门写一个函数
    word_vectors_str = word_vectors_to_str(word_vectors)
    sentences_map = dict(zip(word_vectors_str,sentences))
    #用余弦相似度进行判定的k-means进行聚类
    k_means = KMeans(119)
    k_means.fit(word_vectors)
    #将结果展示出来
    show(k_means.clusters,sentences_map)

