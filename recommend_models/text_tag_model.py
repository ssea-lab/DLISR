# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("..")
from keras.regularizers import l2

from main.new_para_setting import new_Para
from main.dataset import meta_data, dataset
from keras.initializers import Constant


from recommend_models.HIN_sim import mashup_HIN_sims
from Helpers.util import cos_sim, save_2D_list
from recommend_models.recommend_Model import gx_model
import numpy as np
from keras.layers.core import Dropout, Reshape
from keras.layers import Lambda, Concatenate, Add, Embedding, multiply, Multiply
from keras.layers import Dense, Input, AveragePooling2D, concatenate
from keras.models import Model
from keras import backend as K
from main.processing_data import process_data, get_mashup_api_allCategories
import tensorflow as tf
from embedding.encoding_padding_texts import encoding_padding

MAX_SEQUENCE_LENGTH = new_Para.param.MAX_SEQUENCE_LENGTH
MAX_NUM_WORDS = 30000
channel_axis = 1 if K.image_data_format () == 'channels_first' else 3
max_categories_size = 5


class gx_text_tag_model (gx_model):
    """
    同时处理text和tag的结构；新加入feature特征提取器;但不加入MF部分
    """

    def __init__(self):
        """
        :param content_fc_unit_nums: 获取文本部分的feature，和mf部分平级，必须要有
        :param text_fc_unit_nums:只有当merge_manner='final_merge'时需要
        :param tag_fc_unit_nums:  同上
        """
        super (gx_text_tag_model, self).__init__ ()

        self.categories_feature_extracter = None  # 用于提取categories中的特征，和text公用一个embedding层
        if new_Para.param.merge_manner == 'final_merge':
            self.tag_fc_unit_nums = new_Para.param.tag_fc_unit_nums
            self.text_fc_unit_nums = new_Para.param.text_fc_unit_nums

        self.simple_name = 'old_old_function'  # 旧情景，只用功能
        # 模型名只跟架构有关
        self.model_name += 'Category_type:{} tag_manner:{} merge_manner:{}'.format (new_Para.param.Category_type, new_Para.param.tag_manner,
                                                                         new_Para.param.merge_manner)

        if new_Para.param.merge_manner=='final_merge':
            self.model_name += 'text_fc_unit_nums:{} '.format(self.text_fc_unit_nums).replace(',', ' ')
            self.model_name += 'tag_fc_unit_nums:{} '.format(self.tag_fc_unit_nums).replace(',', ' ')

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        return self.simple_name + self.model_name


    def process_texts(self):  # 处理文本，先进行
        self.encoded_texts = encoding_padding (meta_data.descriptions, new_Para.param.remove_punctuation)  # 可得到各文本的encoded形式

    def process_tags(self):
        self.encoded_tags = encoding_padding (meta_data.tags, new_Para.param.remove_punctuation)  # 可得到各文本的encoded形式


    def show_mashup_text_tag_embeddings(self,train_data,mashup_num=10):
        """
        有bug！！！
        在模型训练之后，观察word embedding之后得到的文本和tag矩阵，只想关注padding的0的映射情况
        :return:
        """
        """
        text_tag_middle_model = Model(inputs=[self.model.inputs[0], self.model.inputs[2]],
                                      outputs=[self.embedding_layer.get_output_at(0),
                                               self.embedding_layer.get_output_at(2)])
                                               
        feature_mashup_ids = list(np.unique(mashup_ids))[:mashup_num] # 选几个就可以
        instances_tuple = self.get_instances(feature_mashup_ids, [0] * len(feature_mashup_ids),
                                                     mashup_only=True)
        """
        m_ids,a_ids,slt_a_is = train_data[:-1]
        instances_tuple = self.get_instances(m_ids[:mashup_num],a_ids[:mashup_num],slt_a_is[:mashup_num])  # 第一维去除tag，第二位选择instances个数
        text_tag_middle_model = Model(inputs=[*self.model.inputs],
                                      outputs=[self.model.get_layer('embedding_layer').get_output_at(0)]) #  self.embedding_layer.get_output_at(1)                                              self.embedding_layer.get_output_at(1)


        # 两个三维矩阵，10*150*50
        text_embeddings = text_tag_middle_model.predict([*instances_tuple], verbose=0) # , tag_embeddings

        # 两个看embedding结果的文件
        text_embeddings_path = os.path.join(dataset.crt_ds.model_path,'mashup_text_embeddings.dat')
        # tag_embeddings_path = os.path.join(Para.model_path,'mashup_tag_embeddings.dat')
        for index in range(mashup_num):
            save_2D_list(text_embeddings_path, text_embeddings[index],'w+')
            # save_2D_list(tag_embeddings_path, tag_embeddings[index], 'w+')

    def get_categories_feature_extracter(self):
        """
        跟标签种类和得到向量的方式都有关系
        最准确的平均，需要知道每个项目的标签的数目，应该从模型外部输入，作为一个input！最后再改！！！
        :return:
        """
        if new_Para.param.tag_manner == 'old_average':
            return self.get_categories_old_feature_extracter () # 不讲数目的全平均
        elif new_Para.param.tag_manner == 'new_average':
            return self.get_categories_new_average_feature_extracter () # 考虑tag数目的平均
        elif new_Para.param.tag_manner == 'sum':
            return self.get_categories_sum_feature_extracter () # sum，没法考虑数目
        elif new_Para.param.tag_manner == 'merging':
            return self.get_categories_merging_feature_extracter ()

    def get_categories_old_feature_extracter(self):
        """
        categories_feature的特征提取器    无序  极短  暂定为各个类别词embedding的平均；老写法  有点问题:全部平均，且未考虑非零数目***
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding_layer = self.get_text_embedding_layer ()
            embedded_results = embedding_layer (categories_input)  # 转化为2D
            embedded_sequences = Lambda (lambda x: tf.expand_dims (x, axis=3)) (
                embedded_results)  # tf 和 keras的tensor 不同！！！
            # average sum/size size变量
            embedded_results = AveragePooling2D (pool_size=(MAX_SEQUENCE_LENGTH, 1)) (
                embedded_sequences)  # 输出(None,1,embedding,1)?
            embedded_results = Reshape ((self.word_embedding_dim,), name='categories_feature_extracter') (
                embedded_results)  # 为了能够跟text得到的 (None,1,embedding) merge

            self.categories_feature_extracter = Model (categories_input, embedded_results)
        return self.categories_feature_extracter

    def get_categories_sum_feature_extracter(self):
        """
        categories_feature的特征提取器    无序  极短  暂定为各个类别词embedding的平均；有问题，再改！！！不使用add，使用沿一个轴的相加，除以每个项目的tag数目
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            # size=len(np.nonzero(categories_input)) # 词汇数 对tensor不能使用nonzero！
            embedding_layer = self.get_text_embedding_layer ()
            embedded_results = embedding_layer (categories_input)  # 转化为2D (samples, sequence_length, output_dim)
            print('categories_sum_feature_extracter:')
            print('before reduce_sum:',embedded_results)
            embedded_results = Lambda(lambda x: tf.reduce_sum(x,axis=1))([embedded_results])
            print('after reduce_sum:', embedded_results)
            # embedded_results = Lambda(lambda x: tf.divide(added,size))(added)  # tf 和 keras的tensor 不同！！！可以将向量每个元素除以实数？
            self.categories_feature_extracter = Model (categories_input, embedded_results)
        return self.categories_feature_extracter

    def get_categories_new_average_feature_extracter(self):
        """
        categories_feature的特征提取器    无序  极短  暂定为各个类别词embedding的平均；有问题，再改！！！不使用add，使用沿一个轴的相加，除以每个项目的tag数目
        :return:
        """
        # 之前的仅适用于padding的0的embedding也为0的情况
        if self.categories_feature_extracter is None:
            categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            categories_size_input_reciprocal = Input(shape=(1,), dtype='float32')
            categories_size_input = Lambda(lambda x:1/x)(categories_size_input_reciprocal) # tag的长度 不可以！！
            
            # size=len(np.nonzero(categories_input)) # 词汇数 对tensor不能使用nonzero！
            embedding_layer = self.get_text_embedding_layer ()
            embedded_results = embedding_layer (categories_input)  # 转化为2D (samples, sequence_length, output_dim)

            # 切片且求和
            def slide_sum(paras):
                _embedded_results = paras[0]
                _categories_size_input = paras[1]

                def fn(elements):
                    _embedded_results_ = elements[0]
                    _categories_size_input_ = K.cast (K.squeeze(elements[1],axis=0), tf.int32)
                    #具体原因未知：bug: From merging shape 0 with other shapes. for 'model_2_2/map/while/strided_slice/stack_1' (op: 'Pack') with input shapes: [1], [].
                    if len(_categories_size_input_.shape)==1:
                        _categories_size_input_ = K.cast (K.squeeze(_categories_size_input_,axis=0), tf.int32)
                    # print('_embedded_results_1',_embedded_results_)
                    # print('_categories_size_input_', _categories_size_input_)

                    # def slice2D(x, index):
                    #     return x[150-index:, :]
                    # embedded_results_ = Lambda(slice2D, arguments={'index':_categories_size_input_})(_embedded_results_) # 切片 2D

                    _embedded_results_= _embedded_results_[MAX_SEQUENCE_LENGTH-_categories_size_input_:, :] # 切片  2D
                    # print('_embedded_results_2',_embedded_results_)
                    _embedded_results_ = Lambda(lambda x: K.sum(x, axis=0))(_embedded_results_)
                    # print('_embedded_results_3', _embedded_results_)

                    return _embedded_results_

                return K.map_fn(fn, (_embedded_results, _categories_size_input), dtype=(tf.float32))

            embedded_results = Lambda(slide_sum)([embedded_results, categories_size_input])
            # print('after reduce_sum:', embedded_results)
            embedded_results = Multiply()([embedded_results,categories_size_input_reciprocal])
            # print('after devide,embedded_results:', embedded_results)
            self.categories_feature_extracter = Model(
                inputs=[categories_input, categories_size_input_reciprocal],
                outputs=[embedded_results],name='categories_feature_extracter')

            # print('build new_average_feature_extracter,done!')
        return self.categories_feature_extracter

    def get_categories_merging_feature_extracter(self):
        """
        categories_feature的特征提取器    整合最多三个类别词的embedding
        :return:
        """
        if self.categories_feature_extracter is None:
            categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedding_layer = self.get_text_embedding_layer ()
            embedded_results = embedding_layer (categories_input)

            getindicelayer = Lambda (lambda x: x[:, -1 * max_categories_size:, :])
            embedded_results = getindicelayer (embedded_results)  # 三个词的embedding # (samples, 3, output_dim)
            print ('before mering, shape of sliced embedding :')
            print (embedded_results.shape)

            # results=concatenate([embedded_results[0],embedded_results[1],embedded_results[2]]) # 默认最后一个轴 # (samples, 1, 3*output_dim)
            # bug：AttributeError: 'Tensor' object has no attribute '_keras_history'  直接索引的tensor是tf中tensor

            # 老方法,先切片后拼，没必要
            # getindicelayer1 = Lambda (lambda x: x[:, 0, :])
            # layer1 = getindicelayer1 (embedded_results)
            # print ('before mering,shape:')
            # print (layer1.shape)
            #
            # getindicelayer2 = Lambda (lambda x: x[:, 1, :])
            # getindicelayer3 = Lambda (lambda x: x[:, 2, :])
            # results = Concatenate (name='categories_feature_extracter') (
            #     [layer1, getindicelayer2 (embedded_results), getindicelayer3 (embedded_results)])

            reshaped_dim = int(embedded_results.shape[1].value)*int(embedded_results.shape[2].value)
            results = Reshape((reshaped_dim,),name='cate_merging_reshape')(embedded_results)

            print ('mering 3 embedding,shape:')
            print (results.shape)

            self.categories_feature_extracter = Model (categories_input, results)
        return self.categories_feature_extracter

    def get_text_tag_part(self, user_text_input, item_text_input, user_categories_input, item_categories_input,user_tag_nums_input=None,item_tag_nums_input=None):
        """
        同时处理text和tag
        :param user_text_input:
        :param item_text_input:
        :return:
        """
        user_text_feature = self.feature_extracter_from_texts () (user_text_input)  # (None,embedding)
        item_text_feature = self.feature_extracter_from_texts () (item_text_input)

        if new_Para.param.tag_manner == 'new_average':
            user_categories_feature = self.get_categories_feature_extracter () ([user_categories_input,user_tag_nums_input])
            item_categories_feature = self.get_categories_feature_extracter () ([item_categories_input,item_tag_nums_input])
        else:
            user_categories_feature = self.get_categories_feature_extracter () (user_categories_input)
            item_categories_feature = self.get_categories_feature_extracter () (item_categories_input)

        if new_Para.param.merge_manner == 'direct_merge':
            x = Concatenate (name='concatenate_1') ([user_text_feature, item_text_feature, user_categories_feature,
                                 item_categories_feature])  # 整合文本和类别特征，尽管层次不太一样

        elif new_Para.param.merge_manner == 'final_merge':

            x = concatenate ([user_text_feature, item_text_feature])
            for unit_num in self.text_fc_unit_nums:
                x = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (x)

            y = concatenate ([user_categories_feature, item_categories_feature])
            for unit_num in self.tag_fc_unit_nums:
                y = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (y)

            x = concatenate ([x, y])  # 整合文本和类别特征，尽管层次不太一样

        for unit_num in self.content_fc_unit_nums[:-1]:
            x = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (x)

        x = Dense (self.content_fc_unit_nums[-1], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg), name='text_tag_feature_extracter') (x)

        print ('built text and tag layer, done!')
        return x

    def get_model(self):
        if self.model==None:
            # right part
            user_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                     name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
            item_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')

            user_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='user_categories_input')
            item_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_categories_input')

            if new_Para.param.tag_manner == 'new_average':
                user_tag_nums_input = Input (shape=(1,), dtype='float32', name='user_categories_num_input')
                item_tag_nums_input = Input(shape=(1,), dtype='float32', name='item_categories_num_input')
                x = self.get_text_tag_part(user_text_input, item_text_input, user_categories_input, item_categories_input,user_tag_nums_input,item_tag_nums_input)
            else:
                x = self.get_text_tag_part (user_text_input, item_text_input, user_categories_input, item_categories_input)

            predict_vector = Dropout (0.5) (x)
            predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
                predict_vector)

            if new_Para.param.tag_manner == 'new_average':
                self.model = Model (inputs=[user_text_input, item_text_input, user_categories_input, item_categories_input,user_tag_nums_input,item_tag_nums_input],
                                outputs=[predict_result])
            else:
                self.model = Model (inputs=[user_text_input, item_text_input, user_categories_input, item_categories_input],
                                outputs=[predict_result])

            """
            for layer in self.model.layers:
                print(layer.name)
            print ('built whole model, done!')
            """
        print(np.array(self.text_embedding_layer.get_weights ()).shape)
        print (self.text_embedding_layer.get_weights ()[0][:2])
        return self.model

    def get_instances(self, mashup_id_instances, api_id_instances, mashup_and_api=True):
        """
        根据get_model_instances得到的mashup_id_instances, api_id_instances生成该模型需要的样本
        train和test样例都可用  但是针对一维列表形式，所以test先需拆分！！！

        修改，针对需要平均的tag 特征，需要提供每个样本的tag数目
        :param args:
        :return:
        """

        # mashup/api的类型信息
        mashup_categories = [get_mashup_api_allCategories ('mashup', meta_data.mashup_id2info, mashup_id, new_Para.param.Category_type) for
                             mashup_id in mashup_id_instances]
        api_categories = [get_mashup_api_allCategories ('api', meta_data.api_id2info, new_Para.param, new_Para.param.Category_type) for api_id in
                          api_id_instances]

        mashup_tag_nums = [1/len(api_cates) for api_cates in api_categories]  # 每个样本的tag数目
        api_tag_nums= [1/len(api_cates) for api_cates in api_categories] # 每个样本的tag数目

        if not mashup_and_api: # only mashup
            if new_Para.param.tag_manner == 'new_average':
                examples = (
                    np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
                    np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding')),
                    np.array(mashup_tag_nums)
                )
            else:
                examples = (
                    np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
                    np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding'))
                )
        else:
            if new_Para.param.tag_manner == 'new_average':
                examples = (
                    np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
                    np.array (self.encoded_texts.get_texts_in_index (api_id_instances, 'keras_setting', self.num_users)),
                    np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding')),
                    np.array (self.encoded_texts.get_texts_in_index (api_categories, 'self_padding')),
                    np.array(mashup_tag_nums),
                    np.array(api_tag_nums)
                )
            else:
                examples = (
                    np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
                    np.array (self.encoded_texts.get_texts_in_index (api_id_instances, 'keras_setting', self.num_users)),
                    np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding')),
                    np.array (self.encoded_texts.get_texts_in_index (api_categories, 'self_padding'))
                )
        return examples

    def get_mashup_text_tag_features(self,mashup_ids,save_results=True):
        """
        传入待测mashup的id列表，返回特征提取器提取的mashup的text和tag的feature
        :param mashup_ids: 可以分别传入train和test的mashup
        :return:
        """
        if new_Para.param.tag_manner== 'new_average':
            text_tag_middle_model = Model (inputs=[self.model.inputs[0], self.model.inputs[2], self.model.inputs[4], self.model.inputs[5]],
                                       outputs=[self.model.get_layer ('concatenate_1').input[0],
                                                self.model.get_layer ('concatenate_1').input[2]])
        else:
            text_tag_middle_model = Model (inputs=[self.model.inputs[0], self.model.inputs[2]],
                                       outputs=[self.model.get_layer ('concatenate_1').input[0],
                                                self.model.get_layer ('concatenate_1').input[2]])

        feature_mashup_ids=list(np.unique(mashup_ids))
        feature_instances_tuple = self.get_instances(feature_mashup_ids,[0]*len(feature_mashup_ids), mashup_and_api=False)
        text_features,tag_features=text_tag_middle_model.predict ([*feature_instances_tuple], verbose=0)
        if save_results:
            text_features_path = os.path.join(dataset.crt_ds.model_path.format(self.get_simple_name()), 'mashup_text_features.dat')
            tag_features_path = os.path.join(dataset.crt_ds.model_path.format(self.get_simple_name()),
                                                 'mashup_tag_features.dat')
            save_2D_list(text_features_path, text_features, 'w+')
            save_2D_list(tag_features_path, tag_features, 'w+')
        return text_features,tag_features


# item id 部分还要改动！！！
class gx_text_tag_CF_model (gx_text_tag_model):
    # mf_fc_unit_nums 部分没有用
    # mashup_api_matrix 是U-I 调用矩阵；ini_features_array：mashup的text和tag训练好的整合的特征，初始化使用；max_ks 最大的topK个邻居
    def __init__(self,mf_fc_unit_nums,
                 pmf_01, mashup_api_matrix, u_factors_matrix, i_factors_matrix, i_id_list, ini_features_array, max_ks,
                 predict_fc_unit_nums=[], text_fc_unit_nums=[], tag_fc_unit_nums=[]):

        super (gx_text_tag_CF_model, self).__init__ (mf_fc_unit_nums)

        self.mashup_api_matrix = mashup_api_matrix
        self.u_factors_matrix = u_factors_matrix
        self.i_factors_matrix = i_factors_matrix
        self.i_id_list= tf.constant(i_id_list) # (n,)?
        self.pmf_01 = pmf_01

        self.max_ks = max_ks
        self.max_k = max_ks[-1]  # 一定要小于num——users！

        self.predict_fc_unit_nums = predict_fc_unit_nums  # 用于整合文本和mf之后的预测
        # 在一个batch的计算过程中，如果已经计算过某个mashup最近似的mashup，记录其局部index和sim，下个样本不需要计算；加速。不规范
        # 训练时每个batch之后若feature变化，则需要外部初始化。predict不需要更新

        self.feature_dim=len(ini_features_array[0])
        # self.id2SimMap={} 编码困难，用两个list代替
        self.update_features (ini_features_array)

    # embedding等文本部分参数可变时才有意义，batch更新后进行
    def update_features(self, ini_features_array):
        self.features = tf.Variable (ini_features_array, dtype='float32',
                                     trainable=False)  # 存储所有mashup特征的变量  tf中的tensor  需要在必要处区分keras的tensor
        # self.id2SimMap = {}
        self._userIds = []
        self.con_user_ids = None
        self._topkIndexes = []
        self.stack_topkIndexes = None
        self._topkSims = []
        self.stack_topkSims = None

    def get_model(self, text_tag_model=None):

        user_text_input, item_text_input, user_categories_input, item_categories_input, x, mashup_feature = None, None, None, None, None, None

        if text_tag_model is None:  # 根据结构从头开始搭建模型
            # 文本特征提取部分right part
            user_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='float32',
                                     name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
            item_text_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')
            user_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='user_categories_input')
            item_categories_input = Input (shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_categories_input')

            x = self.get_text_tag_part (user_text_input, item_text_input, user_categories_input,
                                        item_categories_input)  # 整合了mashup，api对的特征
            # print(x.shape)
            mashup_feature = concatenate ([self.user_text_feature,
                                           self.user_categories_feature])  # 有问题，还没改*** get_text_tag_part步骤中获取的mashup的text和tag feature
            # print(mashup_feature.shape)

        else:  # 利用传入的搭建好的模型搭建模型
            user_text_input, item_text_input, user_categories_input, item_categories_input = text_tag_model.inputs

            x = text_tag_model.get_layer ('text_tag_feature_extracter').output  # 获取的文本特征部分
            """
            text_feature=text_tag_model.get_layer('model_1').get_output_at(0)
            tag_feature = text_tag_model.get_layer('model_2').get_output_at(0)
            """
            # 针对的是direct merge的方法，整合用户的特征
            text_feature = text_tag_model.get_layer ('concatenate_1').input[0]
            tag_feature = text_tag_model.get_layer ('concatenate_1').input[2]
            mashup_feature = Concatenate () ([text_feature, tag_feature])

        U_I, u_factors, i_factors = None, None, None
        # CF part
        if self.pmf_01 == '01':
            U_I = K.variable (self.mashup_api_matrix, dtype='int32')
        elif self.pmf_01 == 'pmf':
            u_factors = K.variable (self.u_factors_matrix, dtype='float32')  # 外部调用包，传入  存储
            i_factors = K.variable (self.i_factors_matrix, dtype='float32')

        user_id = Input (shape=(1,), dtype='int32', name='user_input')  # 返回1D tensor,可当做变量的索引
        item_id = Input (shape=(1,), dtype='int32', name='item_input')

        def lam(paras):
            with tf.name_scope ('topK_sim'):
                lam_user_id = paras[0]
                lam_item_id = paras[1]
                lam_mashup_feature = paras[2]

                # 每个样本的数据进行相同的处理
                def fn(elements):
                    # ***为什么搭建时是int32，使用Lambda层传入数据后自动变为float32?***
                    a_user_id = tf.cast (elements[0], tf.int32)  # 用来判断该mashup是否计算过最近邻
                    # 用来获取api的latent factor
                    # a_item_id = tf.squeeze (tf.cast (elements[1], tf.int32))  # scalar # scalar shape: (1,)  要作为索引要转化为()***
                    # 用来判断api是否存在于训练集中
                    a_item_id =tf.cast (elements[1], tf.int32)

                    a_mashup_feature = elements[2]  # 1D tensor

                    def cpt_top_sims():

                        sims = []
                        same_sim_scalar = tf.constant (1.0)
                        small_tr = tf.constant (0.00001)
                        same_scalar = tf.constant (0.0)
                        for index in range (self.features.shape[0]):  # 跟sim有关索引的全部是局部index
                            sim = tensor_sim (a_mashup_feature, self.features[index])
                            final_sim = tf.cond (tf.abs (sim - same_sim_scalar) <= small_tr, lambda: same_scalar,
                                                 lambda: sim)  # 如果输入的feature和历史近似（float近似），认为相等，设为0
                            sims.append (final_sim)  # list of scalar

                        tensor_sims = [K.expand_dims (sim) for sim in sims]
                        tensor_sims = K.concatenate (tensor_sims)  # shape=(n,)
                        # print(tensor_sims.shape)

                        max_sims, max_indexes = tf.nn.top_k (tensor_sims, self.max_k)  # shape=(n,)
                        max_sims = tf.squeeze (max_sims / tf.reduce_sum (max_sims))  # 归一化*** (1,n)->(n,)
                        # self.id2SimMap[a_user_id]=(max_sims,max_indexes)  # scalar(1,) -> (shape=(n,),shape=(n,))

                        self._userIds.append (a_user_id)  # scalar(1,)!!!
                        self.con_user_ids = K.concatenate (self._userIds)  # (n,)
                        self._topkIndexes.append (max_indexes)  # scalar(n,)
                        self._topkSims.append (max_sims)
                        self.stack_topkIndexes = tf.stack (self._topkIndexes)
                        self.stack_topkSims = tf.stack (self._topkSims)
                        print ('this mahsup has never been cpted!')

                        print ('max_sims and max_indexes in cpt_top_sims,')
                        print (max_sims)
                        print (max_indexes)
                        return [max_sims, max_indexes]

                    """
                    def get_top_sims():#
                        temp_returns=K.constant([0]*self.max_k,dtype='int32'),K.constant(np.zeros((self.max_k,)))
                        for temp_user_id in list(self.id2SimMap.keys()):
                            index_sim=tf.cond(tf.equal(temp_user_id, a_user_id),lambda:self.id2SimMap.get(temp_user_id),lambda:temp_returns)
                            if index_sim is not temp_returns:
                                return index_sim
                    """

                    def get_top_sims():  #
                        max_sims = tf.Variable (np.zeros ((self.max_k,)), dtype='float32')
                        max_indexes = tf.Variable (np.zeros ((self.max_k,)), dtype='int32')
                        if len (self._userIds) == 0:  # 不可能
                            return [max_sims, max_indexes]
                        else:
                            index = tf.constant (0)
                            if len (self._userIds) > 1:
                                index = tf.squeeze (
                                    tf.where (tf.equal (self.con_user_ids, a_user_id))[0])  # where:[?,1] -> [1]->()

                            max_sims = self.stack_topkSims[index]  # concatenate 对m个(n,)结果是（m*n,)；而stack是(m,n) [index]后是(n,)
                            max_indexes = self.stack_topkIndexes[index]
                        print ('this mahsup has been cpted!')
                        print ('max_sims and max_indexes in get_top_sims,')
                        print (max_sims)
                        print (max_indexes)
                        return [max_sims, max_indexes]

                    def scalar_in():
                        false = tf.constant (False, dtype='bool')
                        true = tf.constant (True, dtype='bool')
                        if len (self._userIds) == 0:
                            return false
                        elif len (self._userIds) == 1:
                            return tf.squeeze (tf.equal (self._userIds[0], a_user_id))
                        else:
                            # user_ids=K.concatenate(list(K.expand_dims(self.id2SimMap.keys()))) # shape=(n,)
                            is_in = tf.reduce_any (tf.equal (a_user_id, self.con_user_ids))
                            return is_in  # shape=() dtype=bool>

                    # *** scalar用in? 虽然值相同，但是对象不同.使用tf.equal
                    # lambda作为参数？函数callable？
                    max_sims, max_indexes = tf.cond (scalar_in (), get_top_sims, cpt_top_sims)

                    # 判断api id是否存在于训练集中
                    def i_id_In():
                        return tf.reduce_any (tf.equal (a_item_id, self.i_id_list))
                    i_local_index=tf.squeeze (tf.where (tf.equal (self.i_id_list, a_item_id))[0])

                    CF_feature = None
                    if self.pmf_01 == 'pmf':
                        sum_u_factor = K.variable (np.zeros_like (self.u_factors_matrix[0]))
                        for i in range (self.max_k):
                            index = max_indexes[i]
                            temp_sim = max_sims[i]
                            sum_u_factor += temp_sim * u_factors[index]

                        # 获取api的factor  id->局部索引
                        api_factor = tf.cond (i_id_In (),
                                              lambda: i_factors[i_local_index],
                                              tf.Variable (np.zeros((self.feature_dim,)), dtype='float32'))
                        # CF_feature = K.concatenate ([sum_u_factor, i_factors[a_item_id]]) 没注意id映射的错误写法
                        CF_feature = K.concatenate ([sum_u_factor, api_factor]) #

                    elif self.pmf_01 == '01':
                        topK_prod = []

                        for i in range (self.max_k):
                            index = max_indexes[i]
                            temp_sim = max_sims[i]
                            # u_i = U_I[index][a_item_id]
                            u_i=tf.cond (i_id_In (),
                                              lambda: U_I[index][i_local_index],
                                              tf.Variable (0.0, dtype='float32'))
                            topK_prod.append (temp_sim * u_i)

                        topk_sim_features = [K.expand_dims (sum (topK_prod[:topK])) for topK in
                                             self.max_ks]  # 各个topK下计算的sim积  tensor
                        CF_feature = K.concatenate (topk_sim_features)  # 整合的tensor 形状？

                    return a_user_id, a_item_id, CF_feature  # 同时返回user_id是为了保证输入和输出形状相同，user_id无实质意义

                _1, _2, CF_feature = K.map_fn (fn, (lam_user_id, lam_item_id, lam_mashup_feature))

                return CF_feature

        CF_feature = Lambda (lam) ([user_id, item_id, mashup_feature])

        """看是否需要加入MLP后再整合
        for unit_num in self.text_fc_unit_nums:
            CF_feature = Dense(unit_num, activation='relu')(CF_feature)

        for unit_num in self.tag_fc_unit_nums:
            x = Dense(unit_num, activation='relu')(x)
        """
        predict_vector = concatenate ([CF_feature, x])  # 整合文本和类别特征，尽管层次不太一样
        print (predict_vector.shape)

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector)

        # predict_vector = Flatten()(predict_vector)
        predict_vector = Dropout (0.5) (predict_vector)
        predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
            predict_vector)

        self.model = Model (
            inputs=[user_id, item_id, user_text_input, item_text_input, user_categories_input, item_categories_input],
            outputs=[predict_result])

        print ('built whole model, done!')
        return self.model

    def get_name(self):
        name = super (gx_text_tag_CF_model, self).get_name ()
        name += self.pmf_01 + '_'
        name += 'max_ks:{} '.format (self.max_ks).replace (',', ' ')
        name += 'predict_fc_unit_nums:{} '.format (self.predict_fc_unit_nums).replace (',', ' ')
        return 'gx_text_tag_CF_model:' + name  # ***

    def get_instances(self, mashup_id_instances, api_id_instances):
        # mashup/api的类型信息
        mashup_categories = [get_mashup_api_allCategories ('mashup', meta_data.mashup_id2info, mashup_id, new_Para.param.Category_type) for
                             mashup_id
                             in mashup_id_instances]
        api_categories = [get_mashup_api_allCategories ('api', meta_data.api_id2info, api_id, new_Para.param.Category_type) for api_id in
                          api_id_instances]

        # 针对使用预训练的text_tag_model(embedding复用）
        if self.encoded_texts is None:
            self.process_texts ()
        # print('train examples:' + str(len(mashup_id_instances)))
        examples = (
            np.array (mashup_id_instances),
            np.array (api_id_instances),
            np.array (self.encoded_texts.get_texts_in_index (mashup_id_instances, 'keras_setting', 0)),
            np.array (self.encoded_texts.get_texts_in_index (api_id_instances, 'keras_setting', self.num_users)),
            np.array (self.encoded_texts.get_texts_in_index (mashup_categories, 'self_padding')),
            np.array (self.encoded_texts.get_texts_in_index (api_categories, 'self_padding'))
        )

        """
        if api_id_instances[0]!=api_id_instances[-1]: # 不保存feature用例的
            np.savetxt('../data/getInstences_encoding_texts1', examples[2],fmt='%d')
            np.savetxt('../data/getInstences_encoding_texts2', examples[3],fmt='%d')
            np.savetxt('../data/getInstences_encoding_texts3', examples[4],fmt='%d')
            np.savetxt('../data/getInstences_encoding_texts4', examples[5],fmt='%d')
            print('save getInstences_encoding_texts,done!')
        """
        return examples


# 需要改成参数，数据类形式！
class gx_text_tag_only_MLP_model (gx_text_tag_model):
    def __init__(self, mf_fc_unit_nums,
                 u_factors_matrix, i_factors_matrix, m_id2index,i_id2index, ini_features_array, num_feat, topK,CF_self_1st_merge,cf_unit_nums,text_weight=0.5,
                 predict_fc_unit_nums=[],
                 if_co=3, if_pop=False,co_unit_nums=[1024,256,64,16]):

        super (gx_text_tag_only_MLP_model, self).__init__ (mf_fc_unit_nums)

        self.u_factors_matrix = u_factors_matrix
        self.i_factors_matrix = i_factors_matrix
        self.ini_features_array = ini_features_array
        self.m_index2id = {index: id for id, index in m_id2index.items ()}
        self.i_id2index = i_id2index

        # self.max_k = max_ks[-1]  # 一定要小于num——users！
        self.predict_fc_unit_nums = predict_fc_unit_nums

        self.num_feat = num_feat
        self._map = None  # pair->x
        self.x_feature_dim = None
        self.mashup_id2CFfeature = None  # mashup-> text,tag 100D
        self.topK = topK
        self.text_weight=text_weight

        self.CF_self_1st_merge=CF_self_1st_merge
        self.cf_unit_nums=cf_unit_nums
        self.model = None

        self.if_co = if_co # 0是没有，1是跟所有api的共现次数向量；2是跟最近邻mashup调用过的api的共现次数；3是最近邻mashup是否调用过该api，50D
        self.if_pop = if_pop
        self.co_unit_nums=co_unit_nums
        self.api_id2covec,self.api_id2pop = meta_data.pd.get_api_co_vecs()
        self.mashup_id_pair = meta_data.pd.get_mashup_api_pair('dict') # 每个mashup调用的api序列

        self.mashup_id2neighbors={}
        self.mashup_id2CFfeature = {}

    def get_name(self):
        name = super (gx_text_tag_only_MLP_model, self).get_name ()
        cf_='_cf_unit' if self.CF_self_1st_merge  else ''
        name=name+cf_

        co_= '_coInvoke_' + str(self.if_co)
        pop_='_pop_' if self.if_pop else ''

        return 'gx_text_tag_MLP_only_model:' + name+ '_KNN_'+str(self.topK)+'_textWeight_'+str(self.text_weight)+co_+pop_ # ***

    def initialize(self, text_tag_recommend_model, text_tag_model, train_mashup_id_list, train_api_id_list,
                   test_mashup_id_list, test_api_id_list, feature_train_mashup_ids):
        """
        假设mashup的text和tag feature不变，并且衡量mashup相似度的时候，加权text和tag相似度得到最终确定的相似度
        """

        prod = len (test_mashup_id_list) * len (test_mashup_id_list[0])
        D1_test_mashup_id_list = tuple (np.array (test_mashup_id_list).reshape (prod, ))  # 将二维的test数据降维
        D1_test_api_id_list = tuple (np.array (test_api_id_list).reshape (prod, ))

        feature_test_mashup_ids = sorted (list (set (D1_test_mashup_id_list)))  # 测试用mashup的升序排列
        feature_test_api_ids = [0] * len (feature_test_mashup_ids)
        feature_instances_tuple = text_tag_recommend_model.get_instances (feature_test_mashup_ids, feature_test_api_ids,
                                                                          True)  # 只包含mashup信息

        # test样本：提取text和tag feature
        text_tag_middle_model_1 = Model (inputs=[text_tag_model.inputs[0], text_tag_model.inputs[2]],
                                         outputs=[text_tag_model.get_layer ('concatenate_1').input[0],
                                                  text_tag_model.get_layer ('concatenate_1').input[2]])
        text_tag_test_mashup_features = np.hstack (
            text_tag_middle_model_1.predict ([*feature_instances_tuple], verbose=0))  # text，tag 按照mashup id的大小顺序

        # 训练+测试样本  求所有样本的  mashupid，apiid：x
        all_mashup_id_list = train_mashup_id_list + D1_test_mashup_id_list
        all_api_id_list = train_api_id_list + D1_test_api_id_list
        all_instances_tuple = text_tag_recommend_model.get_instances (all_mashup_id_list, all_api_id_list)
        text_tag_middle_model = Model (inputs=text_tag_model.inputs,
                                       outputs=[text_tag_model.get_layer (
                                           'text_tag_feature_extracter').output])  # 输出mashup api的text,tag整合后的特征

        x_features = text_tag_middle_model.predict ([*all_instances_tuple])
        self.x_feature_dim = len (x_features[0])
        self._map = {}  # 基于id
        for index in range (len (x_features)):
            self._map[(all_mashup_id_list[index], all_api_id_list[index])] = x_features[index]

        # 先train，后test mashup id
        all_feature_mashup_ids = feature_train_mashup_ids + feature_test_mashup_ids
        all_features = np.vstack ((self.ini_features_array, text_tag_test_mashup_features))

        # CNN提取的文本特征和tag的embedding大小不一样，所以无法直接拼接计算sim;需要单独计算sim，然后加权求和!!!
        text_dim=self.inception_fc_unit_nums[-1]

        for i in range (len (all_feature_mashup_ids)):  # 为所有mashup找最近
            id2sim = {}
            for j in range (len (feature_train_mashup_ids)):  # 从所有train中找,存放的是内部索引
                if i != j:
                    text_sim = cos_sim (all_features[i][:text_dim], all_features[j][:text_dim])
                    tag_sim = cos_sim (all_features[i][text_dim:], all_features[j][text_dim:])
                    id2sim[j]=  self.text_weight*text_sim+(1- self.text_weight)*tag_sim
            topK_indexes, topK_sims = zip (*(sorted (id2sim.items (), key=lambda x: x[1], reverse=True)[:self.topK]))
            self.mashup_id2neighbors[all_feature_mashup_ids[i]]=[self.m_index2id[index] for index in topK_indexes] #每个mashup距离最近的mashup的id list
            topK_sims = np.array (topK_sims) / sum (topK_sims)
            cf_feature = np.zeros ((self.num_feat))
            for z in range (len (topK_indexes)):
                cf_feature += topK_sims[z] * self.u_factors_matrix[topK_indexes[z]]
            self.mashup_id2CFfeature[all_feature_mashup_ids[i]] = cf_feature

    def get_model(self):
        # 搭建简单模型
        pair_x = Input (shape=(self.x_feature_dim,), dtype='float32')
        mashup_cf = Input (shape=(self.num_feat,), dtype='float32')
        api_cf = Input (shape=(self.num_feat,), dtype='float32')
        co_dim= self.topK if self.if_co == 3 or  self.if_co == 4 else len(self.api_id2covec) # 3：最近邻是否调用 50D
        co_invoke=Input (shape=(co_dim,), dtype='float32')
        pop=Input (shape=(1,), dtype='float32')

        predict_vector = None
        if self.if_co==4: # 只使用content+50D：1+3  再改
            predict_vector = pair_x
            predict_vector1 = Dense (self.co_unit_nums[0], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (co_invoke)
            for unit_num in self.co_unit_nums[1:]:
                predict_vector1=Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector1)
            predict_vector = concatenate ([predict_vector,predict_vector1])
            predict_vector = Dropout (0.5) (predict_vector)
            predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
                predict_vector)

            self.model = Model (inputs=[pair_x, co_invoke], outputs=[predict_result])
            print('只使用content+50D!,模型搭建成功！')
            return self.model

        if self.CF_self_1st_merge:
            predict_vector=concatenate ([mashup_cf, api_cf])
            for unit_num in self.cf_unit_nums:
                predict_vector = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector)
            predict_vector = concatenate ([predict_vector, pair_x])
        else:
            predict_vector = concatenate ([mashup_cf, api_cf, pair_x])  # 整合文本和类别特征，尽管层次不太一样

        if self.if_co:
            predict_vector1 = Dense (self.co_unit_nums[0], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (co_invoke)
            for unit_num in self.co_unit_nums[1:]:
                predict_vector1=Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector1)
            predict_vector = concatenate ([predict_vector,predict_vector1])

        if self.if_pop:
            predict_vector = concatenate ([predict_vector, pop])

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector)
        predict_vector = Dropout (0.5) (predict_vector)
        predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
            predict_vector)

        if not (self.if_co or self.if_pop):
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x],outputs=[predict_result])
        elif self.if_co and not self.if_pop:
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x,co_invoke], outputs=[predict_result])
        elif self.if_pop and not self.if_co:
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x,pop], outputs=[predict_result])
        elif self.if_pop and  self.if_co:
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x,co_invoke,pop], outputs=[predict_result])

        return self.model

    # 需要按照continue model的方法重新搭建一个相似度计算动态变化的模型！
    def get_instances(self, mashup_id_list, api_id_list):
        mashup_cf_features, api_cf_features, x_features, api_co_vecs,api_pops = [], [], [],[],[]
        api_zeros = np.zeros ((self.num_feat))
        num_api=len(self.api_id2covec[0])
        for i in range (len (mashup_id_list)):
            mashup_id = mashup_id_list[i]
            api_id = api_id_list[i]
            mashup_cf_features.append (self.mashup_id2CFfeature[mashup_id])
            api_i_feature = self.i_factors_matrix[self.i_id2index[api_id]] if api_id in self.i_id2index.keys() else api_zeros
            api_cf_features.append (api_i_feature)
            x_features.append (self._map[(mashup_id, api_id)])

            if self.if_co:
                if self.if_co==1:
                    api_co_vecs.append(self.api_id2covec[api_id])
                elif self.if_co==2:
                    api_co_vec=np.zeros((num_api))
                    for m_neigh_id in self.mashup_id2neighbors:
                        for _api_id in self.mashup_id_pair[m_neigh_id]: # 邻居mashup调用过的api
                            api_co_vec[_api_id]=self.api_id2covec[api_id][_api_id]
                    api_co_vecs.append (api_co_vec)
                elif self.if_co == 3 or self.if_co == 4: # 是否被最近邻调用
                    api_co_vec = np.zeros ((self.topK))
                    api_co_vec=[1 if api_id in self.mashup_id_pair[m_neigh_id] else 0 for m_neigh_id in self.mashup_id2neighbors[mashup_id]]
                    api_co_vecs.append (api_co_vec)
            if self.if_pop:
                api_pops.append(self.api_id2pop[api_id])

        if self.if_co == 4:
            return np.array (x_features), np.array (api_co_vecs)
        if not (self.if_co or self.if_pop):
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features)
        elif self.if_co and not self.if_pop:
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features), np.array (api_co_vecs)
        elif self.if_pop and not self.if_co:
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features), np.array (api_pops)
        elif self.if_pop and  self.if_co:
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features),np.array(api_co_vecs),np.array(api_pops)


class gx_text_tag_only_MLP_model_new (gx_text_tag_model):
    """
    修改了原来的模型；
    1.复用function_only模型时，只用mashup，api的feature，重新拼接训练参数，而不是复用拼接MLP之后的pair_x向量
    2.衡量mashup相似度时，改进了原有的只根据mahsup text和tag的计算方法，借助HIN的方法，动态学习各种相似度的权重
    """
    def __init__(self):

        super (gx_text_tag_only_MLP_model_new, self).__init__ ()

        # i_factors_matrix 按照全局id排序; UI矩阵也是按照api id从小到大排序
        # 而mashup不同，按照训练集中的内部索引大小
        self.num_feat = new_Para.param.num_feat  # ???
        self.i_factors_matrix = np.zeros((meta_data.api_num,self.num_feat)) # 待改！！！传参和参数类，数据类之间的平衡
        for id,index in dataset.UV_obj.a_id2index.items():
            self.i_factors_matrix[id]=dataset.UV_obj.a_embeddings[index]
        self.m_index2id = {index: id for id, index in dataset.UV_obj.m_id2index.items ()}
        self.m_id2index = dataset.UV_obj.m_id2index
        self.predict_fc_unit_nums = new_Para.param.predict_fc_unit_nums

        self.topK = new_Para.param.topK # 对模型有影响

        self.CF_self_1st_merge= new_Para.param.CF_self_1st_merge
        self.cf_unit_nums=new_Para.param.cf_unit_nums
        self.model = None

        self.if_co = new_Para.param.if_co # 0是没有，1是跟所有api的共现次数向量；2是跟最近邻mashup调用过的api的共现次数；3是最近邻mashup是否调用过该api，50D
        self.if_pop = new_Para.param.if_pop
        self.co_unit_nums=new_Para.param.shadow_co_fc_unit_nums

        self.mashup_id2neighbors={}
        self.mashup_id2CFfeature = {}

    def get_name(self):
        name = super (gx_text_tag_only_MLP_model, self).get_name ()
        cf_='_cf_unit' if self.CF_self_1st_merge  else ''
        name=name+cf_

        co_= '_coInvoke_' + str(self.if_co)
        pop_='_pop_' if self.if_pop else ''

        return 'gx_text_tag_MLP_only_model:' + name+ '_KNN_'+str(self.topK)+'_textWeight_'+co_+pop_ # ***

    def get_simple_name(self):
        simple_name='old_whole' # 旧情景，全部
        return simple_name

    def set_attribute(self): # ,all_mashup_num,all_api_num,his_m_ids
        self.all_mashup_num = meta_data.mashup_num
        self.all_api_num = meta_data.api_num
        self.his_m_ids = dataset.crt_ds.his_mashup_ids

    def set_mashup_api_features(self,recommend_model):
        """
        在get_model()和get_instances()之前设置
        :param recommend_model: 利用function_only模型获得所有特征向量
        :return:
        """
        self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features = recommend_model.get_mashup_api_features(
            self.all_mashup_num, self.all_api_num)

    def set_embedding_weights(self,text_tag_model):
        self.wordindex2emb = np.squeeze(text_tag_model.get_layer ('embedding_layer').get_weights())  # (9187, 50)
        print(self.wordindex2emb.shape)
        print('the embedding of padding 0:')
        print (self.wordindex2emb[0])

    def build_m_i_matrix(self,train_mashup_id_list, train_api_id_list, train_labels):
        # 根据训练数据构建历史的U-I矩阵,mashup按照内部索引顺序， api按全局id！！！ i_factors也应该与之对应！
        self.m_a_matrix=np.zeros((len(self.his_m_ids),self.all_api_num),dtype='int32')
        for index in range(len(train_mashup_id_list)):
            if  train_labels[index]==1:
                m_index=self.m_id2index[train_mashup_id_list[index]]
                self.m_a_matrix[m_index][train_api_id_list[index]]=1

    def cpt_feature_sims_dict(self,mashup_features):
        # 得到每个mashup到历史mashup的text，tag特征的余弦相似度
        sim_dict={}
        for m_id in range(self.all_mashup_num):
            for his_m_id in self.his_m_ids:
                min_m_id=min(m_id,his_m_id)
                max_m_id =max(m_id,his_m_id)
                sim_dict[(min_m_id,max_m_id)]=cos_sim(mashup_features[min_m_id],mashup_features[max_m_id])
        return sim_dict

    def get_feature_sim(self,sim_dict,m_id1,m_id2):
        if m_id1==m_id2:
            return 0
        else:
            return sim_dict[(min(m_id1,m_id2),max(m_id1,m_id2))]

    def get_model(self):
        # 按照continuemodel的方法重新搭建一个相似度计算动态变化的模型！
        # set the embedding value for the  padding value
        mashup_texts_features = np.vstack((self.mashup_texts_features, np.zeros(1, self.inception_fc_unit_nums[-1])))
        api_texts_features = np.vstack((self.api_texts_features, np.zeros(1, self.inception_fc_unit_nums[-1])))
        mashup_tag_features = np.vstack((self.mashup_tag_features, np.zeros(1, self.word_embedding_dim)))
        api_tag_features = np.vstack((self.api_tag_features, np.zeros(1, self.word_embedding_dim)))

        if self.model is None:
            mashup_id_input = Input(shape=(1,), dtype='int32', name='mashup_id_input')
            api_id_input = Input(shape=(1,), dtype='int32', name='api_id_input')

            m_sims_input = Input(shape=(len(self.m_index2id), 4,), dtype='float32',
                                 name='mashup_sims_input')  # 四种相似度，基于embedding和特征提取器的*text/tag

            def get_func_features_input():
                # get the functional feature input, using the embedding layer instead of packing in the instance, to save memory
                mashup_text_feature_embedding_layer = Embedding(self.all_mashup_num, self.inception_fc_unit_nums[-1],
                                                                embeddings_initializer=Constant(mashup_texts_features),
                                                                mask_zero=False,
                                                                trainable=False,
                                                                name=' mashup_text_feature_embedding_layer')  # input_length=slt_item_num,

                api_text_feature_embedding_layer = Embedding(self.all_api_num, self.inception_fc_unit_nums[-1],
                                                             embeddings_initializer=Constant(api_texts_features),
                                                             mask_zero=False,
                                                             trainable=False,
                                                             name=' api_text_feature_embedding_layer')  # input_length=slt_item_num,

                mashup_tag_feature_embedding_layer = Embedding(self.all_mashup_num, self.word_embedding_dim,
                                                               embeddings_initializer=Constant(mashup_tag_features),
                                                               mask_zero=False,
                                                               trainable=False,
                                                               name=' mashup_tag_feature_embedding_layer')  # input_length=slt_item_num,

                api_tag_feature_embedding_layer = Embedding(self.all_api_num, self.word_embedding_dim,
                                                            embeddings_initializer=Constant(api_tag_features),
                                                            mask_zero=False,
                                                            trainable=False,
                                                            name=' api_tag_feature_embedding_layer')  # input_length=slt_item_num,

                # (?, embedding)
                m_text_features = mashup_text_feature_embedding_layer(mashup_id_input)
                m_tag_features = mashup_tag_feature_embedding_layer(mashup_id_input)
                a_text_features = api_text_feature_embedding_layer(api_id_input)
                a_tag_features = api_tag_feature_embedding_layer(api_id_input)


                _func_features_input = Concatenate()(m_text_features, m_tag_features, a_text_features,a_tag_features)
                return _func_features_input

            func_features_input = get_func_features_input()

            # 相似度乘以权重得到最终相似度值
            m_sims = Dense(1, activation='linear', use_bias=False, kernel_initializer='uniform', name="m_sims")\
                    (m_sims_input)  # (?,his_num,1)

            u_factors = K.variable(dataset.UV_obj.m_embeddings, dtype='float32')  # 外部调用包，传入  存储
            i_factors = K.variable(self.i_factors_matrix,dtype='float32')
            m_a_matrix = K.variable(self.m_a_matrix, dtype='int32')

            def sim_lam(paras):
                _m_sims_input = paras[0]
                _api_id_input = paras[1]

                # 每个样本的数据进行相同的处理
                def fn(elements):
                    a_m_sims = K.squeeze(elements[0])  # 默认消除维度为1的-> (his_num,)
                    a_api_id = K.cast(K.squeeze(elements[1]), tf.int32)  # 待测api id (1,)->()

                    max_sims, max_indexes = tf.nn.top_k(a_m_sims, self.topK)  # (50,),  是历史mashup的内部索引
                    max_sims = max_sims / tf.reduce_sum(max_sims)  # 归一化
                    max_sims = K.reshape(max_sims, (50, 1))

                    neighbor_m_cf_feas = K.gather(u_factors, max_indexes, axis=1)  # (50,25) # 最近邻的cf feature
                    m_cf_feas = tf.reduce_sum(max_sims * neighbor_m_cf_feas,axis=0)  # *:(50,25)  (25,)  该mquery的cf feature
                    a_cf_feas = i_factors[a_api_id]

                    # co_invoke
                    column_vec = K.squeeze(m_a_matrix[:, a_api_id])  # slide:-> (?,1) 最好是(m_num,1)  squeeze:-> (?)
                    co_vec = K.gather(column_vec, max_indexes)  # neighbor 是否调用过该api: (50,)

                    return m_cf_feas, a_cf_feas, co_vec

                return K.map_fn(fn, (_m_sims_input, _api_id_input))

            m_cf_feas, a_cf_feas, co_vecs = Lambda(sim_lam)([m_sims, api_id_input])

            predict_vector = Dense(self.content_fc_unit_nums[0], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg))(func_features_input)
            for unit_num in self.content_fc_unit_nums[1:]:
                predict_vector = Dense(unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg))(predict_vector)

            if self.CF_self_1st_merge:
                predict_vector2 = Concatenate()([m_cf_feas, a_cf_feas])
                for unit_num in self.cf_unit_nums:
                    predict_vector2 = Dense(unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg))(predict_vector2)
                predict_vector = Concatenate()((predict_vector, predict_vector2))

            if self.if_co:  # 显式历史交互
                predict_vector3 = Dense(self.co_unit_nums[0], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg))(co_vecs)
                for unit_num in self.co_unit_nums[1:]:
                    predict_vector3 = Dense(unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg))(predict_vector3)
                predict_vector = Concatenate()((predict_vector, predict_vector3))

            for unit_num in self.predict_fc_unit_nums:
                predict_vector = Dense(unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg))(predict_vector)

            predict_vector = Dropout(0.5)(predict_vector)
            predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
                predict_vector)

            self.model = Model(
                inputs=[mashup_id_input, api_id_input, m_sims_input],
                outputs=[predict_result])

        return self.model

    def get_instances(self, mashup_id_list, api_id_list):
        # features_tuple 根据gx_text_tag_continue_model.get_mashup_api_features()得到
        # his_m_ids 是每个历史mashup id
        # 获取每个mashup，api的特征不在这里做,在占空间;而是放在模型搭建时使用embedding得到.mashup_slt_apis_list最后要padding

        # 基于提取的text和tag特征计算mashup和历史mashup的相似度
        if self.text_sim_dict is None:
            self.text_sim_dict = self.cpt_feature_sims_dict(self.mashup_texts_features)
        if self.tag_sim_dict is None:
            self.tag_sim_dict = self.cpt_feature_sims_dict(self.mashup_tag_features)

        # 每个待测mashup和历史mashup的相似度
        mhs = mashup_HIN_sims(self.wordindex2emb, self.encoded_texts)
        # 某个样本到每个历史mashup的相似度矩阵（训练样本与自身的相似度为0
        mashup_sims_dict = {}  # (mashup_id,mashup_slt_apis_list：sims) 共用
        mashup_HIN_sims_instances = []
        mashup_text_fea_sims_instances = []  # 分别基于text和tag的feature衡量得到的sim
        mashup_tag_fea_sims_instances = []

        for i in range(len(mashup_id_list)):
            m_id = mashup_id_list[i]
            _key = m_id
            if _key not in mashup_sims_dict.keys():
                # 该mashup到每个历史mashup的sim：二维  num*6  按照内部索引的顺序，和UI矩阵中mashup的索引相同
                _value = [mhs.get_mashup_HIN_sims(m_id, his_m_id) if his_m_id != m_id else [0.0] * 2
                    for his_m_id in self.his_m_ids]  # 2D  his_mashups*2
                mashup_sims_dict[_key] = _value
            else:
                _value = mashup_sims_dict[_key]
            mashup_HIN_sims_instances.append(_value)

            mashup_text_fea_sims_instances.append(
                [self.get_feature_sim(self.text_sim_dict, m_id, his_m_id) for his_m_id in self.his_m_ids])
            mashup_tag_fea_sims_instances.append(
                [self.get_feature_sim(self.tag_sim_dict, m_id, his_m_id) for his_m_id in self.his_m_ids])

        mashup_sims_instances = np.hstack((
                                          np.array(mashup_HIN_sims_instances), np.array(mashup_text_fea_sims_instances),
                                          np.array(mashup_tag_fea_sims_instances)))  # (?,his_mashups,8) 6-8-10
        mhs.save_changes()

        return np.array(mashup_id_list), np.array(api_id_list), mashup_sims_instances


# 搭建模型阶段 抽象tensor的运算
def tensor_sim(f1, f2):
    fenmu = K.sum (tf.multiply (f1, f2))
    sum1 = K.sqrt (K.sum (K.square (f1)))
    sum2 = K.sqrt (K.sum (K.square (f2)))
    return fenmu / (sum1 * sum2)
