# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

from collections import OrderedDict
from itertools import chain

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Concatenate, Dense, Embedding, Input, Reshape, add
from tensorflow.python.keras.regularizers import l2

from deepCTR.layers.sequence import SequencePoolingLayer
from deepCTR.layers import Hash


def create_singlefeat_inputdict(feature_dim_dict, prefix=''):
    # 根据提供的属性域，提供对应的模型input
    sparse_input = OrderedDict()
    for i, feat in enumerate(feature_dim_dict["sparse"]):
        sparse_input[feat.name] = Input(
            shape=(1,), name=feat.name, dtype=feat.dtype)  # prefix+'sparse_' + str(i) + '-' + feat.name)

    dense_input = OrderedDict()

    for i, feat in enumerate(feature_dim_dict["dense"]):
        dense_input[feat.name] = Input(
            shape=(1,), name=feat.name)  # prefix+'dense_' + str(i) + '-' + feat.name)

    return sparse_input, dense_input


def create_varlenfeat_inputdict(feature_dim_dict, mask_zero=True):
    sequence_dim_dict = feature_dim_dict.get('sequence', []) # 不存在默认返回[]
    sequence_input_dict = OrderedDict()
    for i, feat in enumerate(sequence_dim_dict):  # 最大长度的输入
        sequence_input_dict[feat.name] = Input(shape=(feat.maxlen,), name='seq_' + str(
            i) + '-' + feat.name, dtype=feat.dtype)

    if mask_zero:
        sequence_len_dict, sequence_max_len_dict = None, None
    else: # 严格把控填充的0？ 需要再传入当前样本的长度input，属性和该属性的最大长度的字典
        sequence_len_dict = {feat.name: Input(shape=(
            1,), name='seq_length' + str(i) + '-' + feat.name) for i, feat in enumerate(sequence_dim_dict)}
        sequence_max_len_dict = {feat.name: feat.maxlen
                                 for i, feat in enumerate(sequence_dim_dict)}
    return sequence_input_dict, sequence_len_dict, sequence_max_len_dict


def create_embedding_dict(feature_dim_dict, embedding_size, init_std, seed, l2_reg, prefix='sparse',
                          seq_mask_zero=True):
    # 对单值和不定长多值的输入建立embedding层的字典
    if embedding_size == 'auto':
        print("Notice:Do not use auto embedding in models other than DCN")
        sparse_embedding = {feat.name: Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                 embeddings_initializer=RandomNormal(
                                                     mean=0.0, stddev=init_std, seed=seed),
                                                 embeddings_regularizer=l2(l2_reg),
                                                 name=prefix + '_emb_' + str(i) + '-' + feat.name) for i, feat in
                            enumerate(feature_dim_dict["sparse"])}
    else:

        sparse_embedding = {feat.name: Embedding(feat.dimension, embedding_size,
                                                 embeddings_initializer=RandomNormal(
                                                     mean=0.0, stddev=init_std, seed=seed),
                                                 embeddings_regularizer=l2(
                                                     l2_reg),
                                                 name=prefix + '_emb_' + str(i) + '-' + feat.name) for i, feat in
                            enumerate(feature_dim_dict["sparse"])}

    if 'sequence' in feature_dim_dict:
        count = len(sparse_embedding) # 除了单值稀疏特征输入之外的多值输入，例如不定长的文本
        sequence_dim_list = feature_dim_dict['sequence']
        for feat in sequence_dim_list:
            # if feat.name not in sparse_embedding:
            if embedding_size == "auto":
                sparse_embedding[feat.name] = Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                        embeddings_initializer=RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(
                                                            l2_reg),
                                                        name=prefix + '_emb_' + str(count) + '-' + feat.name,
                                                        mask_zero=seq_mask_zero)

            else:
                sparse_embedding[feat.name] = Embedding(feat.dimension, embedding_size,
                                                        embeddings_initializer=RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(
                                                            l2_reg),
                                                        name=prefix + '_emb_' + str(count) + '-' + feat.name,
                                                        mask_zero=seq_mask_zero)

            count += 1

    return sparse_embedding


def merge_dense_input(dense_input_, embed_list, embedding_size, l2_reg):
    # 整合dense_input Embedding之后的结果
    dense_input = list(dense_input_.values()) # Input的列表
    if len(dense_input) > 0:
        if embedding_size == "auto":
            print("Notice:Do not use auto embedding in models other than DCN")
            if len(dense_input) == 1:
                continuous_embedding_list = dense_input[0]
            else:
                continuous_embedding_list = Concatenate()(dense_input)
            continuous_embedding_list = Reshape(
                [1, len(dense_input)])(continuous_embedding_list)
            embed_list.append(continuous_embedding_list)

        else:
            # ??? 为什么这么写？？？
            continuous_embedding_list = list(
                map(Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg), ),
                    dense_input)) # [(batch_size, embedding_size),(batch_size, embedding_size)...]
            continuous_embedding_list = list(
                map(Reshape((1, embedding_size)), continuous_embedding_list)) # [(batch_size,1, embedding_size),(batch_size,1, embedding_size)...]
            embed_list += continuous_embedding_list

    return embed_list


def merge_sequence_input(embedding_dict, embed_list, sequence_input_dict, sequence_len_dict, sequence_max_len_dict,
                         sequence_fd_list):
    if len(sequence_input_dict) > 0:
        # 所有变长特征的embedding之后的向量
        sequence_embed_dict = get_varlen_embedding_vec_dict(
            embedding_dict, sequence_input_dict, sequence_fd_list)
        # pooling
        sequence_embed_list = get_pooling_vec_list(
            sequence_embed_dict, sequence_len_dict, sequence_max_len_dict, sequence_fd_list)
        embed_list += sequence_embed_list

    return embed_list # [(batch_size, 1, embedding_size),(batch_size, 1, embedding_size)...]


def get_embedding_vec_list(embedding_dict, input_dict, sparse_fg_list,return_feat_list=(),mask_feat_list=()):
    # Input->Embedding 得到向量
    embedding_vec_list = []
    for fg in sparse_fg_list: # 稀疏特征
        feat_name = fg.name
        if len(return_feat_list) == 0  or feat_name in return_feat_list: # 默认需要所有稀疏特征，或者该特征是所需要的
            if fg.hash_flag: # 特征hash？？？
                lookup_idx = Hash(fg.dimension,mask_zero=(feat_name in mask_feat_list))(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name] #

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list

def get_varlen_embedding_vec_dict(embedding_dict, sequence_input_dict, sequence_fg_list):
    # 变长序列特征 Input->Embedding 得到向量
    varlen_embedding_vec_dict = {}
    for fg in sequence_fg_list: # 所有序列特征
        feat_name = fg.name
        if fg.hash_flag:
            lookup_idx = Hash(fg.dimension, mask_zero=True)(sequence_input_dict[feat_name])
        else:
            lookup_idx = sequence_input_dict[feat_name] # 得到Input
        varlen_embedding_vec_dict[feat_name] = embedding_dict[feat_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_pooling_vec_list(sequence_embed_dict, sequence_len_dict, sequence_max_len_dict, sequence_fd_list):
    # 将变长的多值序列输入embedding后的二维结果池化，注意点：supports_masking，mask，声明准确的当前长度

    # 需要处理变长！输出[(batch_size, 1, embedding_size),(batch_size, 1, embedding_size)...]
    # 要么支持mask，并提供mask的tensor(但是并未提供给call()？推理？)
    if sequence_max_len_dict is None or sequence_len_dict is None:
        return [SequencePoolingLayer(feat.combiner, supports_masking=True)(sequence_embed_dict[feat.name]) for feat in
                sequence_fd_list]
    else: # 要么不支持mask，但需要提供长度输入，函数内根据变长自己生成mask
        return [SequencePoolingLayer(feat.combiner, supports_masking=False)(
            [sequence_embed_dict[feat.name], sequence_len_dict[feat.name]]) for feat in sequence_fd_list]


# 所有非空的Input，写法！！！
def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def get_inputs_embedding(feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, init_std, seed,
                         sparse_input_dict, dense_input_dict, sequence_input_dict, sequence_input_len_dict,
                         sequence_max_len_dict, include_linear, prefix=""):
    # 返回非空input()列表，embedding并整合之后的结果，离散和序列特征映射成1维的线性？
    # deep_emb_list [(batch_size, 1, embedding_size), (batch_size, 1, embedding_size)...]

    # 得到feature.name->Embedding层的字典，稀疏单值和多值
    deep_sparse_emb_dict = create_embedding_dict(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, prefix=prefix + 'sparse')
    # 稀疏特征的处理，得到embedding之后的向量列表
    deep_emb_list = get_embedding_vec_list(
        deep_sparse_emb_dict, sparse_input_dict, feature_dim_dict['sparse'])
    # 加入多值序列embedding之后的向量
    deep_emb_list = merge_sequence_input(deep_sparse_emb_dict, deep_emb_list, sequence_input_dict,
                                         sequence_input_len_dict, sequence_max_len_dict, feature_dim_dict['sequence'])
    # dense输入也要进行embedding！继续整合
    deep_emb_list = merge_dense_input(
        dense_input_dict, deep_emb_list, embedding_size, l2_reg_embedding) # [(batch_size,1, embedding_size),(batch_size,1, embedding_size)...]

    if include_linear: # embedding到一个值？？？原始线性特征，比如wide部分/LR的部分输入？？？
        # embedding之后再整合，稀疏和序列特征，没有dense型特征
        linear_sparse_emb_dict = create_embedding_dict(
            feature_dim_dict, 1, init_std, seed, l2_reg_linear, prefix + 'linear')
        linear_emb_list = get_embedding_vec_list(
            linear_sparse_emb_dict, sparse_input_dict, feature_dim_dict['sparse'])
        linear_emb_list = merge_sequence_input(linear_sparse_emb_dict, linear_emb_list, sequence_input_dict,
                                               sequence_input_len_dict,
                                               sequence_max_len_dict, feature_dim_dict['sequence'])
    else:
        linear_emb_list = None

    inputs_list = get_inputs_list(
        [sparse_input_dict, dense_input_dict, sequence_input_dict, sequence_input_len_dict])
    return inputs_list, deep_emb_list, linear_emb_list


def get_linear_logit(linear_emb_list, dense_input_dict, l2_reg):
    # 将dense输入也进行线性转化，和sparse，序列的线性结果相加并输出结果
    if len(linear_emb_list) > 1:
        linear_term = add(linear_emb_list)
    elif len(linear_emb_list) == 1:
        linear_term = linear_emb_list[0]
    else:
        linear_term = None

    dense_input = list(dense_input_dict.values())
    if len(dense_input) > 0:
        dense_input__ = dense_input[0] if len(
            dense_input) == 1 else Concatenate()(dense_input)
        linear_dense_logit = Dense(
            1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg))(dense_input__)
        if linear_term is not None:
            linear_term = add([linear_dense_logit, linear_term])
        else:
            linear_term = linear_dense_logit

    return linear_term


def preprocess_input_embedding(feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, init_std, seed,
                               create_linear_weight=True):
    # feature_dim_dict是输入数据每个特征属性的信息
    # 返回dense和线性embedding的结果，dense_input的映射，非空输入列表

    # feat.name->Input的字典
    sparse_input_dict, dense_input_dict = create_singlefeat_inputdict(
        feature_dim_dict)
    sequence_input_dict, sequence_input_len_dict, sequence_max_len_dict = create_varlenfeat_inputdict(
        feature_dim_dict)
    # embedding
    inputs_list, deep_emb_list, linear_emb_list = get_inputs_embedding(feature_dim_dict, embedding_size,
                                                                       l2_reg_embedding, l2_reg_linear, init_std, seed,
                                                                       sparse_input_dict, dense_input_dict,
                                                                       sequence_input_dict, sequence_input_len_dict,
                                                                       sequence_max_len_dict, create_linear_weight)

    return deep_emb_list, linear_emb_list, dense_input_dict, inputs_list



