import sys

sys.path.append("..")
from keras.activations import tanh
from main.new_para_setting import new_Para
import tensorflow as tf
import keras.backend as K
from keras import Input, Model
from keras.layers import Multiply, Subtract, Concatenate, Dense, Lambda, Conv1D, Conv2D, Reshape, PReLU, \
    Dropout, Add, Activation, TimeDistributed, multiply  # Softmax,
from keras.layers.core import RepeatVector,Permute

# 原始实现，弃用
def attention(query,key,value,att_hidden_units = (64, 16)):
    # 旧实现，不推荐
    """
    :param query: (None,D)
    :param key: (None,slt_api_num,D)
    :param value: (None,slt_api_num,D) 一般等于key
    :return:
    """

    slt_api_num = key.shape[1].value # shape[-1]不要直接当做实数，type是Dimension！使用value!!!
    # query = Lambda(lambda x: K.repeat_elements(x,slt_api_num,1))(query)
    query = RepeatVector(slt_api_num)(query) # (None,slt_api_num,D)

    outer_prod = Multiply()([query,key])
    sub = Subtract()([query,key])

    att_score = Concatenate(name='att_info_concate')([query,key,outer_prod,sub]) # (None,slt_api_num,4*D)

    # 使用卷积对每个slt api进行相同的操作，分别得到若干个分值：
    #  'channels_last' (None,slt_api_num,4*D,1)-> (None,slt_api_num,1,1) -> (None,slt_api_num,1)
    kernel_size = att_score.shape[-1].value
    att_score = Lambda(lambda x:K.expand_dims(x,axis=3))(att_score)
    att_score = Conv2D(att_hidden_units[0], kernel_size=(1,kernel_size), activation='relu', name='att_fc_{}'.format(1))(att_score)
    # 结果是(None,slt_api_num,1,64)

    for index,unit_num in enumerate(att_hidden_units[1:]): #
        att_score = Conv2D(unit_num,kernel_size=(1,1),activation='relu',name='att_fc_{}'.format(index+2))(att_score) # 2维怎么使用一维dense？ 可以conv
        # print(att_score)

    # h层之后再进行softmax
    att_score = Conv2D(1,kernel_size=(1,1),activation='linear',use_bias=False,name='att_fc_h')(att_score) # (None,slt_api_num,1,1)
    att_score = Reshape((slt_api_num,1))(att_score) # (None,slt_api_num,1)
    # att_score = Softmax(axis=1) (att_score) 有问题！！！

    att_result = Multiply()([value,att_score]) # (None,slt_api_num,D)
    att_result = Lambda(lambda x:tf.reduce_sum(x,axis=1))(att_result) #  (None,D)
    print(att_result)
    return att_result

# 把attention作为一个model，fine-tuning时可训练？
def attention_3d_block(slt_api_num,feature_dim,name='',):
    """
    :param query: (None,D)
    :param key: (None,slt_api_num,D)
    :param value: (None,slt_api_num,D) 一般等于key
    :return:
    """

    # slt_api_num = int(key.shape[1])
    # feature_dim = int(key.shape[2])

    query = Input(shape=(feature_dim,),name=name+'query_input')
    key = Input(shape=(slt_api_num,feature_dim,), name=name + 'key_input')
    value = Input(shape=(slt_api_num,feature_dim,), name=name + 'value_input')

    Repeat_query = RepeatVector(slt_api_num)(query) # (None,slt_api_num,D)
    outer_prod = Multiply()([Repeat_query,key])
    sub = Subtract()([Repeat_query,key])
    att_score = Concatenate(name=name+'att_info_concate')([Repeat_query,key,outer_prod,sub]) # (None,slt_api_num,4*D)

    a = Permute((2, 1))(att_score) # shape=(?, 4*D, slt_api_num)
    a = Dense(slt_api_num, activation='softmax')(a) # shape=(?, 4*D, slt_api_num)   # 每个特征上都做softmax
    a = Lambda(lambda x: K.mean(x, axis=1), name=name+'dim_reduction')(a)  # shape=(?, slt_api_num) # 所有平均得到单个service的权重
    a = RepeatVector(feature_dim)(a) # shape=(?,D,slt_api_num)
    a_probs = Permute((2, 1), name=name+'attention_vec')(a) # shape=(?,slt_api_num,D)
    output_attention_mul = Multiply(name=name+'attention_mul')([value, a_probs])  # shape=(?,slt_api_num, D)
    att_result = Lambda(lambda x: tf.reduce_sum(x, axis=1))(output_attention_mul)  # (None,D)

    model = Model(inputs=[query,key,value],outputs=[att_result],name=name+'attBlock')
    return model

# 原来的实现
# def new_attention_3d_block(slt_api_num,feature_dim,name='',):
#     """
#     :param query: (None,D)
#     :param key: (None,slt_api_num,D)
#     :param value: (None,slt_api_num,D) 一般等于key
#     :return:
#     """
#
#     query = Input(shape=(feature_dim,),name=name+'query_input')
#     key = Input(shape=(slt_api_num,feature_dim,), name=name + 'key_input')
#     value = Input(shape=(slt_api_num,feature_dim,), name=name + 'value_input')
#
#     Repeat_query = RepeatVector(slt_api_num)(query) # (None,slt_api_num,D)
#     outer_prod = Multiply()([Repeat_query,key])
#     sub = Subtract()([Repeat_query,key])
#     att_score = Concatenate(name=name+'att_info_concate')([Repeat_query,key,outer_prod,sub]) # (None,slt_api_num,4*D)
#
#     att_score = Dense(36)(att_score) # (None,slt_api_num,36)
#     att_score = PReLU()(att_score)
#     if 'new_3layer' in new_Para.param.CI_handle_slt_apis_mode:
#         att_score = Dense(16)(att_score) # (None,slt_api_num,16)
#         att_score = PReLU()(att_score)
#
#     # att_score = Dense(1, activation='linear')(att_score)  # (None,slt_api_num,1)
#     att_score = Dense(1)(att_score) # (None,slt_api_num,36) # 最后非线性
#     att_score = PReLU()(att_score)
#
#     att_score = Reshape((slt_api_num,),)(att_score)  # (None,slt_api_num)
#     a_probs = Dense(slt_api_num, activation='softmax')(att_score)  # (None,slt_api_num)
#     a_probs = Reshape((slt_api_num,1),)(a_probs)  # (None,slt_api_num,1)
#
#     output_attention_mul = Multiply(name=name+'attention_mul')([a_probs,value])  # shape=(?,slt_api_num, D)
#     att_result = Lambda(lambda x: tf.reduce_sum(x, axis=1))(output_attention_mul)  # (None,D)
#
#     model = Model(inputs=[query,key,value],outputs=[att_result],name=name+'attBlock')
#     return model

def new_attention_3d_block(slt_api_num,feature_dim,name='',):
    """
    :param query: (None,D)
    :param key: (None,slt_api_num,D)
    :param value: (None,slt_api_num,D) 一般等于key
    :return:
    """

    query = Input(shape=(feature_dim,),name=name+'query_input')
    key = Input(shape=(slt_api_num,feature_dim,), name=name + 'key_input')
    value = Input(shape=(slt_api_num,feature_dim,), name=name + 'value_input')

    Repeat_query = RepeatVector(slt_api_num)(query) # (None,slt_api_num,D)
    att_score = Concatenate(name=name + 'att_info_concate')([Repeat_query, key])  # (None,slt_api_num,2*D) 不加入外积和差效果较好?
    # outer_prod = Multiply()([Repeat_query,key])
    # sub = Subtract()([Repeat_query,key])
    # att_score = Concatenate(name=name+'att_info_concate')([Repeat_query,key,outer_prod,sub]) # (None,slt_api_num,4*D)

    att_score = Dense(36)(att_score) # (None,slt_api_num,36)
    att_score = PReLU()(att_score)
    if 'new_3layer' in new_Para.param.CI_handle_slt_apis_mode:
        att_score = Dense(16)(att_score) # (None,slt_api_num,16)
        att_score = PReLU()(att_score)

    # att_score = Dense(1, activation='linear')(att_score)  # (None,slt_api_num,1)
    att_score = Dense(1)(att_score) # (None,slt_api_num,1) # 最后非线性
    att_score = PReLU()(att_score)

    att_score = Reshape((slt_api_num,),)(att_score)  # (None,slt_api_num)
    a_probs = Dense(slt_api_num, activation='softmax')(att_score)  # (None,slt_api_num) 不需要加这一层dense？？？加上效果好。 一般softmax层也会加上dense参数
    # a_probs = Activation('softmax')(att_score) #
    a_probs = Reshape((slt_api_num,1),)(a_probs)  # (None,slt_api_num,1)

    # # 直接全连接+softmax层，有问题！
    # a_probs = Dense(slt_api_num, activation='softmax')(att_score) # (None,slt_api_num,16)
    # a_probs = Permute((2, 1))(a_probs)

    output_attention_mul = Multiply(name=name+'attention_mul')([a_probs,value])  # shape=(?,slt_api_num, D)
    att_result = Lambda(lambda x: tf.reduce_sum(x, axis=1))(output_attention_mul)  # (None,D)

    model = Model(inputs=[query,key,value],outputs=[att_result],name=name+'attBlock')
    return model

def channel_attention(slt_api_num,feature_dim,hidden_dim=32,name=''): # 128太大
    """

    :param slt_api_num: 已选择的个数  C
    :param feature_dim: 每个选择item的特征维度  C'
    :param hidden_dim: 涉及到attention的隐层的维度 K
    :param name:
    :return:
    """
    query = Input(shape=(feature_dim,),name=name+'query_input') # (?,C')
    key = Input(shape=(slt_api_num,feature_dim,), name=name + 'key_input') # (?,C,C')
    value = Input(shape=(slt_api_num,feature_dim,), name=name + 'value_input') # (?,C,C')

    x = Dense(hidden_dim, activation='linear')(key) # (?,C,K)
    y = Dense(hidden_dim, activation='linear',use_bias=False)(query) # (?,K)
    Repeat_y = RepeatVector(slt_api_num)(y)  # (?,C,K) 填补中间一维
    b = Add()([x,Repeat_y]) # (?,C,K)
    b = Lambda(lambda x: K.tanh(x))(b)
    # b= tanh(b) 函数不是层，不能直接在model中使用
    weights = Dense(1, activation='softmax')(b) # (?,C,1)
    # # 填充weights至跟value相同形状。不需要
    # weights = Lambda(lambda x: tf.squeeze(x, axis=2))(weights) # (?,C)
    # weights = RepeatVector(feature_dim)(weights)  # (?,C',C) 填补中间一维
    # weights = Lambda(lambda x: tf.transpose(x,perm=[0, 2, 1]))(weights) # (?,C,C')
    att_result = Multiply()([value,weights]) # 不必要求严格相同的形状，自动广播
    model = Model(inputs=[query,key,value],outputs=[att_result],name=name+'channelAttBlock')
    return model

def channel_attention2(slt_api_num,feature_dim,hidden_dim=32,name=''): # 128太大
    """
    对每个已选择服务的channel进行attention
    :param slt_api_num: 已选择的个数  C
    :param feature_dim: 每个选择item的特征维度  C'
    :param hidden_dim: 涉及到attention的隐层的维度 K
    :param name:
    :return:
    """
    query = Input(shape=(feature_dim,),name=name+'query_input') # (?,C')
    key = Input(shape=(slt_api_num,feature_dim,), name=name + 'key_input') # (?,C,C')
    value = Input(shape=(slt_api_num,feature_dim,), name=name + 'value_input') # (?,C,C')

    key1 = Lambda(lambda x: tf.transpose(x,perm=[0, 2, 1]))(key) # (?,C',C)
    value1 = Lambda(lambda x: tf.transpose(x,perm=[0, 2, 1]))(value) # (?,C',C)
    x = Dense(hidden_dim, activation='linear')(key1) # (?,C',K)
    y = Dense(hidden_dim, activation='linear',use_bias=False)(query) # (?,K)
    Repeat_y = RepeatVector(feature_dim)(y)  # (?,C',K) 填补中间一维
    b = Add()([x,Repeat_y]) # (?,C',K)
    b = Lambda(lambda x: K.tanh(x))(b)
    weights = Dense(1, activation='softmax')(b) # (?,C',1)
    att_result = Multiply()([value1,weights]) # 不必要求严格相同的形状，自动广播 # (?,C',C)
    att_result = Lambda(lambda x: tf.transpose(x,perm=[0, 2, 1]))(att_result) # (?,C,C')
    model = Model(inputs=[query,key,value],outputs=[att_result],name=name+'channelAttBlock') #
    return model

def new_attention_3d_block_for_transformer(qs, ks, vs): # 函数式
    slt_api_num = ks.get_shape().as_list()[1]
    print('slt_api_num',slt_api_num)
    feature_dim = qs.get_shape().as_list()[-1]
    print('feature_dim',feature_dim)
    # return new_attention_3d_block(slt_api_num,feature_dim)([qs, ks, vs]) # 形状中都是None，传入后repeatvector有bug！
    model_ = new_attention_3d_block(3, 25)
    output = model_([qs, ks, vs])
    return  output # dk


# transfomer中的attention操作,层的形式
class ScaledDotProductAttention():
    def __init__(self, attn_dropout=0.1):
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask=None):  # mask_k or mask_qk
        # d_k = q.get_shape().as_list()[-1] # Nonetype推导不出来
        d_k = 5
        attn = Lambda(lambda x:tf.matmul(x[0], tf.transpose(x[1], [0, 2, 1])))([q, k])
        attn /= d_k ** 0.5

        # temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        # attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)

        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        # output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        attn = Lambda(lambda x: tf.matmul(x[0],x[1]))([attn, v])
        return attn

# MultiHeadAttention的模型形式，以后还是写成层的形式！！！
def MultiHeadAttention(slt_api_num,feature_dim,n_head, dropout=0.1, name='',att_mode='DIN'): # !!! 'DIN'  'transfomer'   单头transfomer的话直接令head=1即可
    d_k = d_v = feature_dim // n_head
    q = Input(shape=(feature_dim,),name=name+'query_input')
    qs = Lambda(lambda x: K.expand_dims(x,axis=1))(q) # (?,1,100)
    k = Input(shape=(slt_api_num,feature_dim,), name=name + 'key_input')
    v = Input(shape=(slt_api_num,feature_dim,), name=name + 'value_input')
    print('q', qs)
    print('v', v)

    qs_layer = Dense(n_head * d_k, use_bias=False)  # 多头的线性映射实现操作
    ks_layer = Dense(n_head * d_k, use_bias=False)
    vs_layer = Dense(n_head * d_v, use_bias=False)

    w_o = TimeDistributed(Dense(feature_dim))
    # 线性转化
    qs = qs_layer(qs)  # [batch_size, len_q, n_head*d_k]
    ks = ks_layer(k)
    vs = vs_layer(v)
    print('qs', qs)
    print('vs', vs)

    # 输入# [batch_size, len_q, n_head * d_k]，输出[n_head * batch_size, len_q, d_k]
    def reshape1(x):
        s = tf.shape(x)
        # print('s',s)
        x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])
        # print('x', x)
        x = tf.transpose(x, [2, 0, 1, 3])
        # print('x', x)
        x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_q, d_k]
        # print('x', x)
        return x

    qs = Lambda(reshape1)(qs)
    ks = Lambda(reshape1)(ks)
    vs = Lambda(reshape1)(vs)

    if att_mode == 'DIN':
        # head = new_attention_3d_block_for_transformer(qs, ks, vs) 有问题！
        # 直接复制代码
        qs = Reshape((d_k,),)(qs)
        qs = RepeatVector(slt_api_num)(qs)
        ks = Reshape((slt_api_num, d_k), )(ks)
        vs = Reshape((slt_api_num, d_k), )(vs)
        print('qs,ks,vs',qs,ks,vs)

        outer_prod = Multiply()([qs, ks])
        sub = Subtract()([qs, ks])
        att_score = Concatenate(name=name + 'att_info_concate')([qs, ks, outer_prod, sub])
        # att_score = Concatenate(name=name + 'att_info_concate')([qs, ks]) # DIN改3

        att_score = Dense(36)(att_score)  # (None,slt_api_num,36)
        att_score = PReLU()(att_score)
        if 'new_3layer' in new_Para.param.CI_handle_slt_apis_mode:
            att_score = Dense(16)(att_score)  # (None,slt_api_num,16)
            att_score = PReLU()(att_score)
        att_score = Dense(1)(att_score)  # (None,slt_api_num,1) # 最后非线性
        att_score = PReLU()(att_score)

        att_score = Reshape((slt_api_num,), )(att_score)  # (None,slt_api_num)
        a_probs = Dense(slt_api_num, activation='softmax')(att_score)  # (None,slt_api_num)
        a_probs = Reshape((slt_api_num, 1), )(a_probs)  # (None,slt_api_num,1)

        output_attention_mul = Multiply(name=name + 'attention_mul')([a_probs, vs])  # shape=(?,slt_api_num, D)
        head = Lambda(lambda x: tf.reduce_sum(x, axis=1))(output_attention_mul)  # (None,D)
        head = Lambda(lambda x: tf.expand_dims(x, axis=1))(head) # (None,1,D)
        print('head',head)

    elif att_mode == 'transfomer':
        # head = ScaledDotProductAttention()(qs, ks, vs, mask=None)

        attn = Lambda(lambda x:tf.matmul(x[0], tf.transpose(x[1], [0, 2, 1])))([qs, ks])
        # 用了效果不太好
        # # 100/4=25  5   0.2
        # def scale(x):
        #     d_k_daoshu = 0.2
        #     tensor_scale = tf.ones_like(x)*d_k_daoshu
        #     x= tf.multiply(x,tensor_scale)
        #     return x
        # attn = Lambda(scale)(attn)

        # temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        # attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
        attn = Activation('softmax')(attn)
        print('attn',attn)
        # attn = dropout(attn)
        # output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        head = Lambda(lambda x: tf.matmul(x[0],x[1]))([attn, vs])
        print('attn', head)

    # 输入[n_head * batch_size, len_v, d_v]，输出[batch_size, len_v, n_head * d_v]
    def reshape2(x):
        s = tf.shape(x)
        x = tf.reshape(x, [n_head, -1, s[1], s[2]])
        x = tf.transpose(x, [1, 2, 0, 3])
        x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
        return x
    head = Lambda(reshape2)(head) # [batch_size, len_v, n_head * d_v]

    # 最后的线性映射
    outputs = w_o(head)   # [batch_size, len_v, d_model]
    outputs = Dropout(dropout)(outputs)
    outputs = Lambda(lambda x: K.squeeze(x,axis=1))(outputs) # 仅存在本应用中

    model = Model(inputs=[q, k, v], outputs=[outputs], name=name + 'MultiHeadAttBlock')
    return model


# # 原始的层的写法
# class MultiHeadAttention():
#     # mode 0 - big martixes, faster; mode 1 - more clear implementation
#     def __init__(self, n_head, d_model, dropout, mode=0,att_mode='DIN'): # att_mode='DIN' 或者是'transfomer'
#         self.mode = mode
#         self.n_head = n_head
#         self.d_k = self.d_v = d_k = d_v = d_model // n_head
#         self.dropout = dropout
#         if mode == 0:
#             self.qs_layer = Dense(n_head * d_k, use_bias=False)  # 多头的线性映射实现操作
#             self.ks_layer = Dense(n_head * d_k, use_bias=False)
#             self.vs_layer = Dense(n_head * d_v, use_bias=False)
#         elif mode == 1:
#             self.qs_layers = []
#             self.ks_layers = []
#             self.vs_layers = []
#             for _ in range(n_head):
#                 self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
#                 self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
#                 self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
#         if att_mode=='DIN':
#             pass # 再补充
#         elif att_mode=='transfomer':
#             self.attention = ScaledDotProductAttention()
#         self.w_o = TimeDistributed(Dense(d_model))
#
#     def __call__(self, q, k, v, mask=None):
#         d_k, d_v = self.d_k, self.d_v
#         n_head = self.n_head
#
#         if self.mode == 0:
#             qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
#             ks = self.ks_layer(k)
#             vs = self.vs_layer(v)
#
#             def reshape1(x):
#                 s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
#                 x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])
#                 x = tf.transpose(x, [2, 0, 1, 3])
#                 x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_q, d_k]
#                 return x
#
#             qs = Lambda(reshape1)(qs)
#             ks = Lambda(reshape1)(ks)
#             vs = Lambda(reshape1)(vs)
#
#             if mask is not None:
#                 mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
#             head, attn = self.attention(qs, ks, vs, mask=mask)
#
#             def reshape2(x):
#                 s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
#                 x = tf.reshape(x, [n_head, -1, s[1], s[2]])
#                 x = tf.transpose(x, [1, 2, 0, 3])
#                 x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
#                 return x
#
#             head = Lambda(reshape2)(head)
#         elif self.mode == 1:
#             heads = [];
#             attns = []
#             for i in range(n_head):
#                 qs = self.qs_layers[i](q)
#                 ks = self.ks_layers[i](k)
#                 vs = self.vs_layers[i](v)
#                 head, attn = self.attention(qs, ks, vs, mask)
#                 heads.append(head);
#                 attns.append(attn)
#             head = Concatenate()(heads) if n_head > 1 else heads[0]
#             attn = Concatenate()(attns) if n_head > 1 else attns[0]
#
#         outputs = self.w_o(head) # 最后的线性映射
#         outputs = Dropout(self.dropout)(outputs)
#         return outputs, attn

if __name__=='__main__':
    query,key,value = Input((25,)),Input((3,25)),Input((3,25))
    print('query,',query)
    print('key,',key)
    # att_result = attention(query,key,value)
    # att_result = attention_3d_block(query, key, value)
    att_result = new_attention_3d_block(3, 25)
