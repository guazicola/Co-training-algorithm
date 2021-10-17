import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
def get_word_vectors(sentences):
        common_words = '的一是不了人我在有这为之大来以上们到说国和地也子道出而要于就下得可你年生自那后能对'
        sentense_word_lists=[]
        for sentence in sentences:
                #去掉常用字
            sentense_word_lists.append([word for word in sentence if word not in common_words])
        sentences = [''.join(sentense_word_list) for sentense_word_list in sentense_word_lists]
        #先对汉语进行分词
        sentences_afterCut = [list(jieba.cut(sentence)) for sentence in sentences]
        sentences_afterSplit = [' '.join(sentence) for sentence in sentences_afterCut]
        #对所有句子进行tf-idf的分析
        tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf.fit(sentences_afterSplit)
        tfidf_transform = tfidf.transform(sentences_afterSplit).toarray()

        return tfidf_transform


def word_vectors_to_str(word_vectors):
        word_vectors_str=[]
        for word_vector in word_vectors:
                words=[]
                for word in word_vector:
                        words.append(str(word))
                word_vectors_str.append(" ".join(words))
        return np.asarray(word_vectors_str)