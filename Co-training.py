import numpy as np
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


class Co_trainingClassifier(object):
    def __init__(self, view1_baseAlgorithm, view2_baseAlgorithm=None, max_iteration=30, buffer_len=75):
        # 初始化用于分别训练两个视图的算法，可以分别指定，也可以都指定一个
        self.view1_baseAlgorithm = view1_baseAlgorithm
        if view2_baseAlgorithm == None:
            # 这个地方的copy.copy很重要，把两个算法的地址分开
            self.view2_baseAlgorithm = copy.copy(view1_baseAlgorithm)
        # 迭代次数，默认30次，因为论文默认的30次
        self.max_iteration = max_iteration
        self.buffer_len = buffer_len

    def fit(self, X_view1, X_view2, y):
        # 将他们转化为数组，因为列表的索引不能为列表，数组的索引可以为列表
        X_view1 = np.asarray(X_view1)
        X_view2 = np.asarray(X_view2)
        y = np.asarray(y)

        # 将标签分为两个视图
        y_view1_set = copy.copy(y)
        y_view2_set = copy.copy(y)

        # 统计有标签数据的下标，并将下标组成一个列表
        label_index_set = [y_index for y_index, y_data in enumerate(y) if y_data != -1]
        label_index_view1_set = copy.copy(label_index_set)
        label_index_view2_set = copy.copy(label_index_set)
        # 统计无标签数据的下标，并将下标组成一个列表
        unlabel_index_set = [y_index for y_index, y_data in enumerate(y) if y_data == -1]

        # 根据有标签数据中正例和反例的比例，定义每次迭代从缓冲池中取多少个正例和反例
        p_labled_quantity = len([y[label_index] for label_index in label_index_set if y[label_index] == 1])
        n_labled_quantity = len([y[label_index] for label_index in label_index_set if y[label_index] == 0])
        p_quantity = -1
        n_quantity = -1
        if p_labled_quantity > n_labled_quantity:
            n_quantity = 5
            p_quantity = round(n_quantity*(p_labled_quantity / n_labled_quantity))
        else:
            p_quantity = 5
            n_quantity = round(p_quantity*(n_labled_quantity / p_labled_quantity))
        # 第一次取缓冲区
        # 将无标签数据的下标列表打乱
        random.shuffle(unlabel_index_set)
        # 取后buffer_len个数据的下标构成缓冲区。为什么不是前面的数据？因为最后要用pop函数
        buffer_index_set = unlabel_index_set[-min(len(unlabel_index_set), self.buffer_len):]
        # 将缓冲区的下标从无标签数据下标中剔除出去
        unlabel_index_set = unlabel_index_set[:-min(len(unlabel_index_set), self.buffer_len)]

        # 将两个视图的各种东西进行打包，以便参与下面的循环
        # 基础算法打包
        view_baseAlgorithms = [self.view1_baseAlgorithm, self.view2_baseAlgorithm]
        # 将标签打包
        y_view_sets = [y_view1_set, y_view2_set]
        # 将标签打包,这里是反着的，先2后1
        y_view_sets_reverse = [y_view2_set, y_view1_set]
        # 视图数据打包
        X_views = [X_view1, X_view2]
        # 视图中预测概率值打包，前反后正
        y_view_proba_sets = [[], []]
        # 缓冲池中选出的正例数据的下标集合和反例数据的下标集合打包
        p_view_index_sets = [[], []]
        n_view_index_sets = [[], []]
        # 将已标签数据的下标进行打包
        label_index_view_sets = [label_index_view1_set, label_index_view2_set]
        # 将已标签数据进行打包,这里是反着的，先2后1
        label_index_view_sets_reverse = [label_index_view2_set, label_index_view1_set]

        # 迭代训练，当达到最大迭代次数或者无标签数据量为0时，跳出迭代
        iteration = 0
        while iteration < self.max_iteration and unlabel_index_set:
            print('迭代次数：', iteration)
            iteration += 1
            # 将所有需要分视图的多个变量进行挨个的遍历
            # j的顺序是1，2，i的顺序是2，1
            for viewj_baseAlgorithm, X_viewj, y_viewj_proba_set, p_viewj_index_set, n_viewj_index_set, y_viewj_set, y_viewi_set_reverse, label_index_viewj_set, label_index_viewi_set_reverse \
                    in zip(view_baseAlgorithms, X_views, y_view_proba_sets, p_view_index_sets, n_view_index_sets, y_view_sets, y_view_sets_reverse, label_index_view_sets, label_index_view_sets_reverse):
                # 用baseAlgorithm分别对两个视图的有标签数据进行学习
                viewj_baseAlgorithm.fit(X_viewj[label_index_viewj_set], y_viewj_set[label_index_viewj_set])
                print('view comleted')
                # 用训练好的算法对缓冲区中的无标签数据进行概率性训练
                y_viewj_proba_set = viewj_baseAlgorithm.predict_proba(X_viewj[buffer_index_set])

                # 找出正例概率最高的数据的下标
                # argsort()将数组排序之后返回元素下标
                for i in (y_viewj_proba_set[:, 0].argsort())[:n_quantity]:
                    n_viewj_index_set.append(buffer_index_set[i])
                # 找出反例概率最高的数据的下标
                for i in (y_viewj_proba_set[:, 1].argsort())[:p_quantity]:
                    p_viewj_index_set.append(buffer_index_set[i])

                # 在已标签数据中加入数据和标签
                # 反向加入数据
                label_index_viewi_set_reverse.extend(p_viewj_index_set)
                label_index_viewi_set_reverse.extend(n_viewj_index_set)
                # 反向加入标签
                for i in p_viewj_index_set:
                    y_viewi_set_reverse[i] = 1
                for i in n_viewj_index_set:
                    y_viewi_set_reverse[i] = 0

                # 从缓冲区中将已经取出的数据剔除
                buffer_index_set = [i for i in buffer_index_set if not (i in p_viewj_index_set or i in n_viewj_index_set)]

                #初始化p，n变量
                p_view_index_sets = [[], []]
                n_view_index_sets = [[], []]
            # 更新缓冲区
            double_p_n = 2 * (p_quantity + n_quantity)
            i = 0
            while i < double_p_n and unlabel_index_set:
                buffer_index_set.append(unlabel_index_set.pop())
                i += 1

    def ifprobe(self, algorithm, X):
        try:
            algorithm.predict_proba([X])
        except:
            return False
        else:
            return True

    def predict(self, X_view1_set, X_view2_set):
        # 预测
        y_view1_set = self.view1_baseAlgorithm.predict(X_view1_set)
        y_view2_set = self.view2_baseAlgorithm.predict(X_view2_set)

        # 判断两个基算法是否都支持predict_proba:
        if_probe = (self.ifprobe(self.view1_baseAlgorithm, X_view1_set[0]) and self.ifprobe(self.view2_baseAlgorithm, X_view2_set[0]))
        #定义最终的标签数组，初始化为-1
        # 为什么是数组？因为库中那些算法的源码就是用的数组
        y_allview_set = np.asarray(len(X_view1_set) * [-1])
        # 将不同视图的两个标签进行合并
        for i, (y_view1_data, y_view2_data) in enumerate(zip(y_view1_set, y_view2_set)):
            # 如果标签一样
            if y_view1_data == y_view2_data:
                y_allview_set[i] = y_view1_data
            # 如果标签不一样，且基算法支持概率预测
            elif if_probe:
                # 计算视图不同分类的平均数，选可能性大的
                # 因为只有一行数据包含在数组中，所以需要用[0]取出来
                y_proba_view1_set = self.view1_baseAlgorithm.predict_proba([X_view1_set[i]])[0]
                y_proba_view2_set = self.view2_baseAlgorithm.predict_proba([X_view2_set[i]])[0]
                y_proba_sum_set = list(y_proba_view1_set + y_proba_view2_set)
                y_proba_max = max(y_proba_sum_set)
                # 因为标签值刚好和下标值对应，所以直接将下标值用作标签值
                y_allview_set[i] = y_proba_sum_set.index(y_proba_max)
            # 如果标签不一样，且基算法不支持概率预测
            else:
                # 随机指定
                y_allview_set[i] = random.randint(0, 1)

        # 确定标签列表里已经全部都打上了标签
        assert not (-1 in y_allview_set)

        return y_allview_set


if __name__ == '__main__':
    # 25000个样本
    N_SAMPLE = 25000
    # 1000个特征
    N_FEATURE = 1000
    # 创建好数据集
    X, y = make_classification(N_SAMPLE, N_FEATURE)
    df = pd.DataFrame(X)

    # 分离出测试集
    X_test_set = X[-N_SAMPLE // 4:]
    y_test_set = y[-N_SAMPLE // 4:]

    # 分离出无标签集
    y[:N_SAMPLE // 2] = -1
    X_unlable_set = X[:N_SAMPLE // 2]
    y_unlable_set = y[:N_SAMPLE // 2]

    # 分离出标签集
    X_lable_set = X[-N_SAMPLE // 2:-N_SAMPLE // 4]
    y_lable_set = y[-N_SAMPLE // 2:-N_SAMPLE // 4]

    # 从X中将测试集分离出去
    X = X[:-N_SAMPLE // 4]
    y = y[:-N_SAMPLE // 4]

    # 协同训练需要用到的，两个视图，前面无标签，后面有标签
    X_view1 = X[:, :N_FEATURE // 2]
    X_view2 = X[:, -N_FEATURE // 2:]


    # -----------------------协同训练算法-----------------------------
    ctc = Co_trainingClassifier(GaussianNB())
    ctc.fit(X_view1, X_view2, y)
    y_ctc_predict_set = ctc.predict(X_test_set[:, :N_FEATURE // 2], X_test_set[:, N_FEATURE // 2:])
    print('co-training classifier report:')
    report_ctc = classification_report(y_test_set, y_ctc_predict_set)
    print(report_ctc)

    # # -----------------------随机森林算法-----------------------------
    rfc = RandomForestClassifier()
    # 学习
    rfc.fit(X_lable_set[:, :N_FEATURE // 2], y_lable_set)
    # 预测
    y_rfc_predict_set = rfc.predict(X_test_set[:, :N_FEATURE // 2])
    # 生成性能报告
    print('random forest classifier report:')
    report_rfc = classification_report(y_test_set, y_rfc_predict_set)
    print(report_rfc)
    # -----------------------贝叶斯算法-----------------------------
    gn = GaussianNB()
    #学习
    gn.fit(X_lable_set[:, :N_FEATURE // 2],y_lable_set)
    #预测
    y_gn_predict_set=gn.predict(X_test_set[:, :N_FEATURE // 2])
    #生成性能报告
    print('GaussianNB report:')
    report_gn = classification_report(y_test_set, y_gn_predict_set)
    print(report_gn)
