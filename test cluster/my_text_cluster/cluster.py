import random
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class KMeans():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.clusters = [[] for i in range(self.n_clusters)]

    def fit(self, X):
        # 随机指定center
        self.cluster_centers_ = random.sample(list(X), self.n_clusters)

        self.iteration = 1
        self.starttime = datetime.now()
        self._updata_clusters(X)
        while self._updata_center() and self.iteration < 10:
            self.iteration += 1
            self._updata_clusters(X)

    def _updata_clusters(self, X):
        self.clusters = [[] for i in range(self.n_clusters)]
        for x in X:
            # 因为cosine_similarity只能处理二维以上数组，所以要reshape
            similarity_x_center = lambda center: cosine_similarity(x.reshape(1, -1), center.reshape(1, -1))[0]
            center = max(self.cluster_centers_, key=similarity_x_center)
            # center在self.cluster_centers_中对应的下标
            index = 0
            for i in self.cluster_centers_:
                if (i == center).all():
                    index_center = index
                index += 1
            self.clusters[index_center].append(x)
        print(self.iteration, ':updata clusters success----', datetime.now() - self.starttime)

    def _updata_center(self):
        new_centers = []
        for cluster in self.clusters:
            new_center = [np.average(i) for i in zip(*cluster)]
            new_centers.append(new_center)
        print(self.iteration, ':updata centers success----', datetime.now() - self.starttime)
        new_centers = np.asarray(new_centers)
        if (new_centers == np.asarray(self.cluster_centers_)).all():
            return False
        else:
            self.cluster_centers_ = new_centers
            return True
