# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Lambda, AveragePooling2D, Reshape, AveragePooling1D,Input,Dense,Flatten
from tensorflow.python.keras.regularizers import l2
from deepCTR.input_embedding import preprocess_input_embedding, get_linear_logit
from main.new_para_setting import new_Para
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_fun
from deepCTR.utils import check_feature_config_dict



def DeepFM(feature_dim_dict, embedding_size=8,
           use_fm=True, dnn_hidden_units=(128, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0,
           init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, linear_emb_list, dense_input_dict, inputs_list = preprocess_input_embedding(feature_dim_dict,
                                                                                               embedding_size,
                                                                                               l2_reg_embedding,
                                                                                               l2_reg_linear, init_std,
                                                                                               seed,
                                                                                               create_linear_weight=True)

    linear_logit = get_linear_logit(linear_emb_list, dense_input_dict, l2_reg_linear) # 各种输入线性转化并相加的结果

    fm_input = concat_fun(deep_emb_list, axis=1) # 各种特征进行embedding之后再拼接 (?,X,embedding_size)
    deep_input = tf.keras.layers.Flatten()(fm_input) # (?,X*embedding_size)
    fm_out = FM()(fm_input)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                   dnn_use_bn, seed)(deep_input)
    deep_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(deep_out)

    if len(dnn_hidden_units) == 0 and use_fm == False:  # only linear
        final_logit = linear_logit
    elif len(dnn_hidden_units) == 0 and use_fm == True:  # linear + FM
        final_logit = tf.keras.layers.add([linear_logit, fm_out])
    elif len(dnn_hidden_units) > 0 and use_fm == False:  # linear +　Deep
        final_logit = tf.keras.layers.add([linear_logit, deep_logit])
    elif len(dnn_hidden_units) > 0 and use_fm == True:  # linear + FM + Deep
        final_logit = tf.keras.layers.add([linear_logit, fm_out, deep_logit])
    else:
        raise NotImplementedError

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def simple_DeepFM(CI_feature_num=4, NI_feature_num=2,CI_feature_dim=50,NI_feature_dim=25,slt_nums=new_Para.param.slt_item_num,
                  final_feature_dim = 32,use_fm=True, dnn_hidden_units=(128, 32), l2_reg_linear=0.00001, l2_reg_embedding=0, l2_reg_dnn=0,
           init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):
    # 原本模型容量主要在embedding?现在相当于无embedding，需要训练的参数主要在线性和DNN中，FM中无参数
    # CI的50维，NI的25维，添加转化层
    CI_feature_list = [Input(shape=(CI_feature_dim,), dtype='float32',name='CI_feature_{}'.format(i)) for i in range(CI_feature_num)]
    NI_feature_list = [Input(shape=(NI_feature_dim,), dtype='float32',name='NI_feature_{}'.format(i)) for i in range(NI_feature_num)]

    slt_text_features = Input(shape=(slt_nums,CI_feature_dim), dtype='float32', name='slt_text_feature')
    slt_tag_features = Input(shape=(slt_nums, CI_feature_dim), dtype='float32', name='slt_tag_feature')
    slt_NI_features = Input(shape=(slt_nums, NI_feature_dim), dtype='float32', name='slt_NI_feature')
    slt_api_features = [slt_text_features,slt_tag_features,slt_NI_features]

    def pooling(slt_apis_feas): # (?,3,16) -< (?,16)
        # x = Lambda(lambda x: tf.expand_dims(x, axis=3))(slt_apis_feas)  # (?,3,16,1)
        # x = AveragePooling2D(pool_size=(slt_nums, 1))(x)  #

        slt_apis_pooling = Lambda(lambda tensor_: tf.reduce_mean(tensor_, axis=1))(slt_apis_feas)  # (?,16)
        # slt_apis_pooling = Lambda(lambda tensor_: tf.reduce_mean(tensor_, axis=1))(slt_apis_feas)  # (?,16)
        # x = AveragePooling1D(pool_size=slt_nums)(slt_apis_feas) # AveragePooling2D
        # slt_apis_pooling = Lambda(lambda tensor_: tf.squeeze(tensor_,axis=1))(x) # (?,16)
        return slt_apis_pooling

    feature_list = CI_feature_list+NI_feature_list+slt_api_features

    input_num = len(feature_list)
    embedding_layers = [Dense(final_feature_dim,activation=None, use_bias=False, kernel_regularizer=l2(l2_reg_embedding),name='embedding_layer_{}'.format(i)) for i in range(input_num)]
    dense_features = [layer(feature) for layer,feature in zip(embedding_layers,feature_list)] #  每一个输入特征进行转化
    dense_features[-3:] = list(map(pooling,dense_features[-3:])) # 此前slt_apis_feas是(?,3,25/50)，pooling

    # 升维 [(?,1,final_feature_dim),(?,1,final_feature_dim)...]
    feature_list_3D = [Reshape((1, final_feature_dim))(feature) for feature in dense_features]

    # FM部分,要使用FM，需要保证特征的维度都相同！！！#
    fm_input = concat_fun(feature_list_3D, axis=1) # 各种特征拼接 (?,X,final_feature_dim)
    fm_out = FM()(fm_input)

    # deep部分
    deep_input = Flatten()(fm_input)  # (?,feature_num*final_feature_dim)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,dnn_use_bn, seed)(deep_input)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)

    # 线性部分 使用原始特征!!!
    linear_input =  feature_list[:-3]+list(map(pooling,feature_list[-3:])) # 对已选择的原始输入先pooling
    linear_input = concat_fun(linear_input, axis=1)
    linear_logit = Dense(1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg_linear))(linear_input)

    if len(dnn_hidden_units) == 0 and use_fm == False:  # only linear
        final_logit = linear_logit
    elif len(dnn_hidden_units) == 0 and use_fm == True:  # linear + FM
        final_logit = tf.keras.layers.add([linear_logit, fm_out])
    elif len(dnn_hidden_units) > 0 and use_fm == False:  # linear +　Deep
        final_logit = tf.keras.layers.add([linear_logit, deep_logit])
    elif len(dnn_hidden_units) > 0 and use_fm == True:  # linear + FM + Deep
        final_logit = tf.keras.layers.add([linear_logit, fm_out, deep_logit])
    else:
        raise NotImplementedError
    output = Dense(2, activation='softmax', name="prediction")(final_logit)
    model = Model(inputs=feature_list, outputs=output)
    return model