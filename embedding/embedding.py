# -*- coding:utf-8 -*-
import os

import gensim
import numpy as np
from main.new_para_setting import new_Para

def get_word_embedding(embedding, embedding_name, word,dimension,initize='random'):
    """
    得到一个词的embedding向量，不存在为0，或者按-切割再相加
    :param embedding:
    :param embedding_name:
    :param word:
    :param dimension:
    :param initize 可以选择不存在词的初始方式 随机化还是0
    :return:
    """
    if embedding_name == 'glove':
        vector = embedding.get(word)
    elif embedding_name == 'google_news':
        vector = embedding.get_vector(word)

    if vector is not None: # 存在该词
        sum_vec = vector
    else:
        if initize== 'random':
            sum_vec = np.random.uniform(-0.25, 0.25, dimension)
        elif initize== 'zero':
            sum_vec = np.zeros(dimension)
        """
        sum_vec = np.zeros(dimension)
        subs = word.split('-')  # 用-连接的，vector累加***
        valid_subs=0
        if len(subs) > 1:
            for sub_word in subs:
                sub_vector = get_word_embedding(embedding, embedding_name, sub_word,dimension)
                if sub_vector is None:  # 子字符串不存在
                    continue
                valid_subs+=1
                sum_vec = sum_vec + sub_vector
            if valid_subs==0:
                valid_subs=1
            sum_vec = sum_vec/valid_subs
            
        if initize == 'random' and (valid_subs>0): # 如果不存在且切割后的词仍不存在
            # sum_vec=np.random.random(dimension)
            sum_vec =np.random.uniform (-0.25, 0.25, dimension)
        """
    return sum_vec


def get_embedding(embedding_name, dimension=50):
    """
    读取预训练的embedding模型
    :param embedding_name:
    :param dimension:
    :return:
    """
    if embedding_name == 'google_news':
        embedding = gensim.models.KeyedVectors.load_word2vec_format(new_Para.param.google_embedding_path, binary=True)
    elif embedding_name == 'glove':
        embedding = {}
        with open(os.path.join(new_Para.param.glove_embedding_path,
                               "glove.6B.{}d.txt".format(dimension)), encoding='utf-8') as f:  # dict: word->embedding(array)
            for line in f:
                values = line.split()
                word = values[0]
                embedding[word] = np.asarray(values[1:], dtype='float32')
    print('Found %s word vectors in pre_trained embedding.' % len(embedding))
    return embedding


def get_embedding_matrix(word2index, embedding_name, dimension=50):
    """
    得到特定语料库词典对应的embedding矩阵
    :param word2index: 本任务语料中的词
    :param embedding_name:
    :param embedding_path:
    :param dimension:
    :return: 2D array
    """
    embedding=get_embedding(embedding_name, dimension)
    # construct an embedding_matrix
    num_words = len(word2index)+1  # 实际词典大小 +1 !!!(很重要！！！ embedding:out = K.gather(self.embeddings, inputs)
    embedding_matrix = np.zeros((num_words, dimension))
    embedding_matrix[0] =np.random.uniform(-0.25, 0.25, dimension) # padding用的0要随机!!!
    for word, index in word2index.items(): # keras 文本预处理后得到的字典 按词频 对单词编码
        embedding_matrix[index]=get_word_embedding(embedding, embedding_name, word, dimension)
    return embedding_matrix


# 有问题！
"""
RuntimeError: Graph disconnected: cannot obtain value for tensor Tensor("input_1:0", shape=(?, 150), dtype=int32) at layer "input_1". The following previous layers were accessed without issue: []
"""
def print_middle_embedding_result(text_tag_rec_model):
    """
    打印embedding的中间层信息
    :param text_tag_rec_model: 推荐模型
    :param instances: 某个api的标签信息
    :return:
    """
    text_tag_embedding_middle_model = Model (inputs=[text_tag_rec_model.get_model().inputs[0],text_tag_rec_model.get_model().inputs[1],text_tag_rec_model.get_model().inputs[2],text_tag_rec_model.get_model().inputs[3]],
                                   outputs=[text_tag_rec_model.embedding_layer.get_output_at(0)])
    instances=text_tag_rec_model.get_instances(Para.train_mashup_id_list[:1], Para.train_api_id_list[:1]) # 2个样本就好
    results=text_tag_embedding_middle_model.predict([*instances], verbose=0)
    print(results)
