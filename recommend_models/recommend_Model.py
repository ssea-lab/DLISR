# -*- coding:utf-8 -*-
import sys

sys.path.append("..")


from embedding.encoding_padding_texts import encoding_padding
from recommend_models.baseline import get_default_gd
from recommend_models.sequence import SequencePoolingLayer


from main.new_para_setting import new_Para
from main.dataset import meta_data, dataset
import numpy as np


from tensorflow.python.keras.layers import Lambda, Concatenate, MaxPooling2D, LSTM, Bidirectional, PReLU, BatchNormalization
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Embedding, concatenate, Multiply
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras import initializers, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam

from embedding.embedding import get_embedding_matrix
from Helpers.cpt_Sim import get_sims_dict
from main.processing_data import process_data
from recommend_models.simple_inception import inception_layer
import tensorflow as tf

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 30000
channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
categories_size=3


class recommend_Model(object):
    """
    共同基类
    """
    def __init__(self):
        """
        再改，因为user和item也不是model本身的属性？
        """
        self.num_users = meta_data.mashup_num
        self.num_items = meta_data.api_num
        self.model = None
        self.inception_MLP_layer = None
        self.model_name_path = ''

    def get_name(self):
        return ''

    # 类别如何处理？增加一部分？
    def get_model(self):
        """
        **TO OVERIDE**
        :return:  a model
        """
        pass

    def get_merge_MLP(self,input1,input2,MLP_layers):
        """
        难点在于建立model的话，需要设定Input，其中要用到具体形状
        """
        pass

    def get_mf_MLP(self,input_dim1,input_dim2,output_dim,MLP_layers):
        """
        返回id-embedding-merge-mlp的model
        """
        # Input Layer
        user_input = Input(shape=(1,), dtype='int32')
        item_input = Input(shape=(1,), dtype='int32')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=input_dim1, output_dim=output_dim,
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=input_dim2, output_dim=output_dim,
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        # MF part
        mf_user_latent = tf.keras.layers.Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = tf.keras.layers.Flatten()(MF_Embedding_Item(item_input))  # why Flatten？
        mf_vector = concatenate([mf_user_latent, mf_item_latent])

        for idx in range(len(MLP_layers)):   # 学习非线性关系
            layer = Dense(MLP_layers[idx],  activation='relu')
            mf_vector = layer(mf_vector)
        model = Model(inputs=[user_input,item_input],outputs=mf_vector)
        return model

    def get_instances(self):
        """
        **TO OVERIDE**
        """
        pass

    def save_sth(self):
        pass


class gx_model(recommend_Model):
    """
    抽象基类  只含处理文本
    """

    def __init__(self):
        """
        embedding时需要使用dataset下的所有语料，所以一个base_dir决定一个实例；而数据集划分则是不同的训练实例，跟train有关，跟模型本身无关
        :param base_dir:
        :param embedding_name: 'glove' or 'google_news'
        :param embedding_dim: word embedding 维度
        :param inception_channels: inception 几种 filter的个数；通道数
        :param inception_pooling: inception 最顶层 pooling的形式
        :param inception_fc_unit_nums:  inception之后FC的设置，最后一个决定整个textCNN的特征维度
        :param content_fc_unit_nums: 特征提取部分 FC设置，决定最终维度（整合各部分文本信息
        :param predict_fc_unit_nums: 最后FC设置
        """
        super(gx_model, self).__init__()

        self.encoded_texts= None # encoding_padding对象  首次得到embedding层时会初始化 # 把encoded_texts和模型分开？？？
        self.encoded_tags = None

        self.gd = None # 使用gensim中的主题模型提取文本特征
        self.text_embedding_matrix = None
        self.text_embedding_layer = None
        self.tag_embedding_matrix = None
        self.tag_embedding_layer = None

        self.word_embedding_dim= new_Para.param.embedding_dim

        self.inception_channels=new_Para.param.inception_channels
        self.inception_pooling=new_Para.param.inception_pooling
        self.inception_fc_unit_nums = new_Para.param.inception_fc_unit_nums

        self.content_fc_unit_nums = new_Para.param.content_fc_unit_nums  # 在没有使用tag时，可用于整合两个text之后预测***

        self.text_feature_extracter=None # 文本特征提取器
        self.mashup_text_feature_extracter = None # mashup和api使用不同的特征提取器时，比如HDP
        self.api_text_feature_extracter = None

        self.model_name = 'pairwise_'+str(new_Para.param.margin) if new_Para.param.pairwise else 'pointwise'
        self.model_name += '_TrainEmbedding:{}'.format(new_Para.param.embedding_train)
        self.model_name += '_EmbeddingL2_{}'.format(new_Para.param.embeddings_regularizer)
        self.model_name += 'remove_punctuation:{} '.format(new_Para.param.remove_punctuation)
        self.model_name +='{}{} '.format(new_Para.param.embedding_name, self.word_embedding_dim)
        self.model_name += '_{}_ '.format(new_Para.param.text_extracter_mode)
        self.model_name += 'inception_channels:{} '.format(self.inception_channels).replace(',', ' ')
        self.model_name += 'inception_pooling:{} '.format(self.inception_pooling)
        self.model_name += 'inception_fc_unit_nums:{} '.format(self.inception_fc_unit_nums).replace(',', ' ')
        self.model_name += 'incepMLPDropout:{}'.format(new_Para.param.inception_MLP_dropout)
        self.model_name += 'incepMLPBN:{}'.format(new_Para.param.inception_MLP_BN)
        self.model_name += 'content_fc_unit_nums:{} '.format(self.content_fc_unit_nums).replace(',', ' ')


    def get_name(self):
        return self.model_name  # *** 用于区别每个模型  应包含选用的embedding，是否使用tag，inception结构，MF结构，总体结构（FC nums）

    def process_texts(self):
        self.encoded_texts = encoding_padding(meta_data.descriptions,new_Para.param.remove_punctuation)  # 可得到各文本的encoded形式


    def process_tags(self):
        self.encoded_tags = encoding_padding (meta_data.tags, new_Para.param.remove_punctuation)  # 可得到各文本的encoded形式


    def get_text_embedding_layer(self, nonstatic=True):
        """"
        得到定制的embedding层

        paras:
        data_dirs: 存放mashup api 信息的文件夹
        embedding_name：使用哪种pre-trained embedding，google_news or glove
        embedding_path:embedding 文件存放路径
        EMBEDDING_DIM：维度
        nonstatic：基于pre-trained embedding是否微调？
        """

        if self.text_embedding_layer is None:
            if self.encoded_texts is None:
                self.process_texts()
            # 得到词典中每个词对应的embedding
            num_words = min(MAX_NUM_WORDS, len(self.encoded_texts.word2index))+ 1  # 实际词典大小 +1  因为0代表0的填充向量
            self.text_embedding_matrix = get_embedding_matrix(self.encoded_texts.word2index, new_Para.param.embedding_name,
                                                              dimension=self.word_embedding_dim)
            print('built embedding matrix, done!')

            self.text_embedding_layer = Embedding(num_words,
                                                  self.word_embedding_dim,
                                                  embeddings_initializer=Constant(self.text_embedding_matrix),
                                                  embeddings_regularizer= regularizers.l2(new_Para.param.embeddings_regularizer),
                                                  input_length=MAX_SEQUENCE_LENGTH,
                                                  mask_zero=True,
                                                  trainable=new_Para.param.embedding_train, name = 'text_embedding_layer')  # 定义一层 # mask_zero=True, !!!

            print('built text embedding layer, done!')
        return self.text_embedding_layer

    def get_tag_embedding_layer(self, nonstatic=True):
        """"
        同text，处理tags,得到定制的embedding层
        """
        if self.tag_embedding_layer is None:
            if self.encoded_tags is None:
                self.process_tags()
            # 得到词典中每个词对应的embedding
            num_words = min(MAX_NUM_WORDS, len(self.encoded_tags.word2index)) + 1  # 实际词典大小 +1  因为0代表0的填充向量
            self.tag_embedding_matrix = get_embedding_matrix(self.encoded_tags.word2index, new_Para.param.embedding_name,
                                                              dimension=self.word_embedding_dim)
            print('built tag embedding matrix, done!')

            self.tag_embedding_layer = Embedding(num_words,
                                                  self.word_embedding_dim,
                                                  embeddings_initializer=Constant(self.tag_embedding_matrix),
                                                  embeddings_regularizer=regularizers.l2(
                                                  new_Para.param.embeddings_regularizer),
                                                  input_length=MAX_SEQUENCE_LENGTH,
                                                  mask_zero=True,
                                                  trainable=new_Para.param.embedding_train,
                                                  name='tag_embedding_layer')  #

            print('built tag embedding layer, done!')
        return self.tag_embedding_layer

    def textCNN_feature_extracter_from_texts(self,embedded_sequences):
        """
        对embedding后的矩阵做textCNN处理提取特征
        :param embedded_sequences:
        :return:
        """

        filtersize_list = [3, 4, 5]
        number_of_filters_per_filtersize = new_Para.param.textCNN_channels # 跟50D接近   #[128,128,128]
        pool_length_list = [2, 2, 2]

        conv_list = []
        for index, filtersize in enumerate(filtersize_list):
            nb_filter = number_of_filters_per_filtersize[index]
            pool_length = pool_length_list[index]
            conv = Conv2D(nb_filter=nb_filter, kernel_size=(filtersize,self.word_embedding_dim), activation='relu')(embedded_sequences)
            pool = MaxPooling2D(pool_size=(pool_length,1))(conv)
            print('a feature map size:', pool)
            flatten = tf.keras.layers.Flatten()(pool)
            conv_list.append(flatten)

        if (len(filtersize_list) > 1):
            out = Concatenate(axis=-1)(conv_list)
        else:
            out = conv_list[0]

        return out

    def LSTM_feature_extracter_from_texts(self,embedded_sequences):
        out=Bidirectional(LSTM(new_Para.param.LSTM_dim))(embedded_sequences)
        # out = LSTM(new_Para.param.LSTM_dim)(embedded_sequences)
        return out

    def SDAE_feature_extracter_from_texts(self):
        pass

    def HDP_feature_extracter_from_texts(self,mashup_api):
        if self.gd is None:
            self.gd = get_default_gd(tag_times=0,mashup_only=True,strict_train=True) # 用gensim处理文本,文本中不加tag
            self.mashup_features ,self.api_features =self.gd.model_pcs(new_Para.param.text_extracter_mode) #
            self.feature_dim = len(self.mashup_features[0])

        ID_input = Input(shape=(1,), dtype='int32')
        if mashup_api=='mashup':
            if self.mashup_text_feature_extracter is None:  # 没求过
                mashup_text_embedding_layer = Embedding(self.all_mashup_num,self.feature_dim,
                                                 embeddings_initializer=Constant(self.mashup_features),
                                                 mask_zero=False, input_length=1,
                                                 trainable=False, name = 'mashup_text_embedding_layer')
                x = mashup_text_embedding_layer(ID_input)
                self.mashup_text_feature_extracter = Model(ID_input, x, name='mashup_text_feature_extracter')
            return self.mashup_text_feature_extracter

        elif mashup_api=='api':
            if self.api_text_feature_extracter is None:  # 没求过
                api_text_feature_extracter = Embedding(self.all_api_num,self.feature_dim,
                                                        embeddings_initializer=Constant(self.api_features),
                                                        mask_zero=False, input_length=1,
                                                        trainable=False, name='api_text_embedding_layer')
                x = api_text_feature_extracter(ID_input)
                self.api_text_feature_extracter = Model(ID_input, x, name='api_text_feature_extracter')

            return self.api_text_feature_extracter

    # 暂时没用
    def get_inception_MLP_layer(self, channel_num,name = ''):
        """
        textCNN/inception后面加MLP的处理，结构之后的最后一个层命名name='text_feature_extracter'
        :param channel_num: 输入形状的最后一维，
        :param name: 可以使用不同的MLP对mashup，api和slt_apis分别进行转化，声明name即可
        :return:
        """

        if self.inception_MLP_layer is None:
            if new_Para.param.text_extracter_mode != 'LSTM' and new_Para.param.if_inception_MLP:
                input = Input(shape=(channel_num,), dtype='float32')
                x = input
                for FC_unit_num in self.inception_fc_unit_nums:
                    x = Dense(FC_unit_num, kernel_regularizer=l2(new_Para.param.l2_reg))(x)  # 默认activation=None
                    if new_Para.param.inception_MLP_BN:
                        x = BatchNormalization(scale=False)(x)
                    x = PReLU()(x)  #
                    if new_Para.param.inception_MLP_dropout:
                        x = tf.keras.layers.Dropout(0.5)(x)
                self.inception_MLP_layer = Model(input, x, name='text_feature_extracter'+name)
        return self.inception_MLP_layer

    def feature_extracter_from_texts(self,mashup_api=None):
        """
        # 更改：把MLP去掉
        对mashup，service的description均需要提取特征，右路的文本的整个特征提取过程
        公用的话应该封装成新的model！
        :param x:
        :return: 输出的是一个封装好的model，所以可以被mashup和api公用
        """
        if new_Para.param.text_extracter_mode=='HDP' and mashup_api is not None:
            return self.HDP_feature_extracter_from_texts(mashup_api)

        if self.text_feature_extracter is None: #没求过时
            text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            text_embedding_layer = self.get_text_embedding_layer()  # 参数还需设为外部输入！
            text_embedded_sequences = text_embedding_layer(text_input)  # 转化为2D

            if new_Para.param.text_extracter_mode in ('inception','textCNN'): # 2D转3D,第三维是channel
                # print(text_embedded_sequences.shape)
                text_embedded_sequences = Lambda(lambda x: tf.expand_dims(x, axis=3))(text_embedded_sequences)  # tf 和 keras的tensor 不同！！！
                print(text_embedded_sequences.shape)

            if new_Para.param.text_extracter_mode=='inception':
                x = inception_layer(text_embedded_sequences, self.word_embedding_dim, self.inception_channels, self.inception_pooling)  # inception处理
                print('built inception layer, done!')
            elif new_Para.param.text_extracter_mode=='textCNN':
                x = self.textCNN_feature_extracter_from_texts(text_embedded_sequences)
            elif new_Para.param.text_extracter_mode=='LSTM':
                x = self.LSTM_feature_extracter_from_texts(text_embedded_sequences)
            else:
                raise ValueError('wrong extracter!')
            print('text feature after inception/textCNN/LSTM model,',x) # 观察MLP转化前，模块输出的特征

            for FC_unit_num in self.inception_fc_unit_nums:
                x = Dense(FC_unit_num, kernel_regularizer=l2(new_Para.param.l2_reg))(x)  # , activation='relu'
                if new_Para.param.inception_MLP_BN:
                    x = BatchNormalization(scale=False)(x)
                x = PReLU()(x)  #
                if new_Para.param.inception_MLP_dropout:
                    x = tf.keras.layers.Dropout(0.5)(x)

            self.text_feature_extracter=Model(text_input, x,name='text_feature_extracter')
        return self.text_feature_extracter

    def get_categories_feature_extracter(self):
        """
        跟标签种类和得到向量的方式都有关系
        最准确的平均，需要知道每个项目的标签的数目，应该从模型外部输入，作为一个input！最后再改！！！
        :return:
        """

        if new_Para.param.tag_manner == 'new_average':
            return self.get_categories_new_average_feature_extracter()  # 考虑tag数目的平均
        elif new_Para.param.tag_manner in ('mean','sum'):
            return self.sequence_pooling()

    def sequence_pooling(self):
        if self.categories_feature_extracter is None:
            categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            categories_size_input_reciprocal = Input(shape=(1,), dtype='float32')

            embedding_layer = self.get_embedding_layer()
            # mask = K.not_equal(categories_input, 0)
            # mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), 'float32'))(categories_input)

            embedded_results = embedding_layer(categories_input)  # 转化为3D (?,150,50)
            print(embedded_results)
            vec = SequencePoolingLayer(new_Para.param.tag_manner, supports_masking=True)(embedded_results)
            # vec = SequencePoolingLayer('mean', supports_masking=False)([embedded_results,categories_size_input_reciprocal])
            self.categories_feature_extracter = Model (inputs = [categories_input,categories_size_input_reciprocal], outputs=[vec])
        return self.categories_feature_extracter

    def get_categories_new_average_feature_extracter(self):
        """
        categories_feature的特征提取器    无序  极短  暂定为各个类别词embedding的平均；有问题，再改！！！不使用add，使用沿一个轴的相加，除以每个项目的tag数目
        :return:
        """
        # 之前的仅适用于padding的0的embedding也为0的情况
        if self.categories_feature_extracter is None:
            categories_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            categories_size_input_reciprocal = Input(shape=(1,), dtype='float32')
            categories_size_input = Lambda(lambda x: 1 / x)(categories_size_input_reciprocal)  # tag的长度 不可以！！

            # size=len(np.nonzero(categories_input)) # 词汇数 对tensor不能使用nonzero！
            embedding_layer = self.get_embedding_layer()
            embedded_results = embedding_layer(categories_input)  # 转化为2D (samples, sequence_length, output_dim)

            # 切片且求和
            def slide_sum(paras):
                _embedded_results = paras[0]
                _categories_size_input = paras[1]

                def fn(elements):
                    _embedded_results_ = elements[0]
                    _categories_size_input_ = K.cast(K.squeeze(elements[1], axis=0), tf.int32)
                    # 具体原因未知：bug: From merging shape 0 with other shapes. for 'model_2_2/map/while/strided_slice/stack_1' (op: 'Pack') with input shapes: [1], [].
                    if len(_categories_size_input_.shape) == 1:
                        _categories_size_input_ = K.cast(K.squeeze(_categories_size_input_, axis=0), tf.int32)
                    # print('_embedded_results_1',_embedded_results_)
                    # print('_categories_size_input_', _categories_size_input_)

                    # def slice2D(x, index):
                    #     return x[150-index:, :]
                    # embedded_results_ = Lambda(slice2D, arguments={'index':_categories_size_input_})(_embedded_results_) # 切片 2D

                    _embedded_results_ = _embedded_results_[MAX_SEQUENCE_LENGTH - _categories_size_input_:, :]  # 切片  2D
                    # print('_embedded_results_2',_embedded_results_)
                    _embedded_results_ = Lambda(lambda x: K.sum(x, axis=0))(_embedded_results_)
                    # print('_embedded_results_3', _embedded_results_)

                    return _embedded_results_

                return K.map_fn(fn, (_embedded_results, _categories_size_input), dtype=(tf.float32))

            embedded_results = Lambda(slide_sum)([embedded_results, categories_size_input])
            # print('after reduce_sum:', embedded_results)
            embedded_results = Multiply()([embedded_results, categories_size_input_reciprocal])
            # print('after devide,embedded_results:', embedded_results)
            self.categories_feature_extracter = Model(
                inputs=[categories_input, categories_size_input_reciprocal],
                outputs=[embedded_results], name='categories_feature_extracter')

            # print('build new_average_feature_extracter,done!')
        return self.categories_feature_extracter


    def get_text_tag_part(self):
        """
        整合文本和tag
        :return:
        """
        pass

    def get_model(self):
        pass

    def get_instances(self,mashup_id_instances, api_id_instances):
        """
        根据get_model_instances得到的mashup_id_instances, api_id_instances生成该模型需要的样本
        train和test样例都可用  但是针对一维列表形式，所以test先需拆分！！！
        :param args:
        :return:
        """
        pass


class DHSR_model(recommend_Model):
    def __init__(self,old_new='old',slt_num=0):
        super(DHSR_model, self).__init__()
        self.mf_fc_unit_nums = new_Para.param.DHSR_layers1
        self.mf_embedding_dim = new_Para.param.mf_embedding_dim
        self.sims_dict = get_sims_dict(False,True) # 相似度对象，可改参数？
        self.sim_feature_size=new_Para.param.sim_feature_size
        self.final_MLP_layers = new_Para.param.DHSR_layers2
        if old_new=='old' and slt_num ==0:
            self.simple_name = 'DHSR'
        else:
            self.simple_name = 'DHSR_{}_{}'.format(old_new,slt_num)

        self.model_name = '_mf_fc_unit_nums:{} '.format(self.mf_fc_unit_nums).replace(',', ' ')
        self.model_name += '_mfDim{}_ '.format(self.mf_embedding_dim)
        self.model_name += 'final_MLP_layers:{} '.format(self.final_MLP_layers).replace(',', ' ')
        self.model_name += '_simSize{}_ '.format(self.sim_feature_size)
        self.model_dir = dataset.crt_ds.model_path.format(self.get_simple_name())  # 模型路径

        self.lr = new_Para.param.CI_learning_rate # 内容部分学习率
        self.optimizer = Adam(lr=self.lr)

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        return self.simple_name+self.model_name

    def get_model(self):
        # Input Layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')
        text_input = Input(shape=(self.sim_feature_size,), dtype='float32', name='text_input')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=self.num_users, output_dim=self.mf_embedding_dim, name='mf_embedding_user',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=self.mf_embedding_dim, name='mf_embedding_item',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        # MF part
        mf_user_latent = tf.keras.layers.Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = tf.keras.layers.Flatten()(MF_Embedding_Item(item_input))  # why Flatten？
        mf_vector = concatenate([mf_user_latent, mf_item_latent])  # element-wise multiply    ???

        for idx in range(len(self.mf_fc_unit_nums)):   # 学习非线性关系
            layer = Dense(self.mf_fc_unit_nums[idx],  activation='relu', name="layer%d" % idx)
            mf_vector = layer(mf_vector)

        # Text part
        # text_input = Dense(10, activation='relu', kernel_regularizer=l2(0.01))(text_input)  #   sim? 需要再使用MLP处理下？

        # Concatenate MF and TEXT parts
        predict_vector = concatenate([mf_vector, text_input])

        for idx in range(len(self.final_MLP_layers)):   # 整合后再加上MLP？
            layer = Dense(self.final_MLP_layers[idx], activation='relu')# name="layer%d"  % idx
            predict_vector = layer(predict_vector)

        predict_vector = tf.keras.layers.Dropout(0.5)(predict_vector)    # 使用dropout?

        if new_Para.param.final_activation == 'softmax':
            predict_vector = Dense(2, activation='softmax', name="prediction")(predict_vector)
        elif new_Para.param.final_activation == 'sigmoid':
            predict_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        # # Final prediction layer
        # predict_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(inputs=[user_input, item_input, text_input],outputs=predict_vector)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances,if_Train=False, test_phase_flag=False):
        sims=[]
        for i in range(len(mashup_id_instances)):
            sim = self.sims_dict.get_mashup_api_sim(mashup_id_instances[i], api_id_instances[i])
            sims.append(sim)

        examples = (np.array(mashup_id_instances),np.array(api_id_instances),np.array(sims))
        return examples

    def save_sth(self):
        self.sims_dict.save_sims_dict()


class DHSR_noMF(DHSR_model):
    def get_name(self):
        return '_DHSR_noMF'

    def get_model(self):
        # Input Layer
        text_input = Input(shape=(self.sim_feature_size,), dtype='float32', name='text_input')

        predict_vector= Dense(self.final_MLP_layers[0], activation='relu')(text_input)

        for idx in range(len(self.final_MLP_layers))[1:]:   # 整合后再加上MLP？
            layer = Dense(self.final_MLP_layers[idx], activation='relu')# name="layer%d"  % idx
            predict_vector = layer(predict_vector)

        predict_vector = tf.keras.layers.Dropout(0.5)(predict_vector)    # 使用dropout?

        # Final prediction layer
        predict_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(inputs=text_input,outputs=predict_vector)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances):
        sims=[]
        for i in range(len(mashup_id_instances)):
            sim = self.sims_dict.get_mashup_api_sim(mashup_id_instances[i], api_id_instances[i])
            sims.append(sim)

        returns=[]
        returns.append(sims)
        return np.array(returns)


class NCF_model(recommend_Model):
    def __init__(self):
        super(NCF_model, self).__init__()
        self.mf_fc_unit_nums = new_Para.param.NCF_layers
        self.mf_embedding_dim = new_Para.param.mf_embedding_dim
        self.reg_layers=new_Para.param.reg_layers
        self.reg_mf=new_Para.param.reg_mf
        self.name = '_NCF'

    def get_model(self):
        num_layer = len(self.layers)  # Number of layers in the MLP
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=self.num_users, output_dim=self.mf_embedding_dim, name='mf_embedding_user',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(self.reg_mf), input_length=1) #

        MF_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=self.mf_embedding_dim, name='mf_embedding_item',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(self.reg_mf), input_length=1) #

        MLP_Embedding_User = Embedding(input_dim=self.num_users, output_dim=int(self.mf_fc_unit_nums[0] / 2), name="mlp_embedding_user",
                                       embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                       embeddings_regularizer=l2(self.reg_layers[0]), input_length=1) #

        MLP_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=int(self.mf_fc_unit_nums[0] / 2), name='mlp_embedding_item',
                                       embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                       embeddings_regularizer=l2(self.reg_layers[0]), input_length=1) #

        # MF part
        mf_user_latent = tf.keras.layers.Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = tf.keras.layers.Flatten()(MF_Embedding_Item(item_input))
        #   mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply
        mf_vector=Multiply()([mf_user_latent, mf_item_latent])

        # MLP part
        mlp_user_latent = tf.keras.layers.Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = tf.keras.layers.Flatten()(MLP_Embedding_Item(item_input))
        #   mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

        for idx in range(1, num_layer):
            layer = Dense(self.mf_fc_unit_nums[idx],  activation='relu', name="layer%d" % idx) # kernel_regularizer=l2(reg_layers[idx]),
            mlp_vector = layer(mlp_vector)

        # Concatenate MF and MLP parts
        # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
        # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
        #   predict_vector = merge([mf_vector, mlp_vector], mode='concat')
        predict_vector = Concatenate()([mf_vector, mlp_vector])

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(input=[user_input, item_input],output=prediction)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances):
        examples = (np.array(mashup_id_instances),np.array(api_id_instances))
        return examples


