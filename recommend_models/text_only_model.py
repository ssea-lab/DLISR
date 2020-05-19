# -*- coding: utf-8 -*-
import heapq
import os
import pickle
import sys

from keras.regularizers import l2

from Helpers.util import save_2D_list, cos_sim
from main.dataset import meta_data, dataset
from main.new_para_setting import new_Para
from recommend_models.recommend_Model import gx_model
import numpy as np
from keras.layers.core import Dropout
from keras.layers import Dense, Input, concatenate, Concatenate
from keras.models import Model
from keras import backend as K

from embedding.encoding_padding_texts import encoding_padding

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 30000
channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
categories_size=3


class gx_text_only_model(gx_model):
    """"只处理text 不处理 tag的结构;但不加入MF部分"""

    def __init__(self):
        super(gx_text_only_model, self).__init__()
        self.simple_name = 'text_only'
        self.model = None

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        return self.simple_name + self.model_name

    def process_texts(self):
        """
        只处理文档  不处理tag
        :param data_dirs: 某个路径下的mashup和api文档
        :return:
        """
        self.encoded_texts=encoding_padding(meta_data.descriptions,new_Para.param.remove_punctuation) # 可得到各文本的encoded形式

    def get_text_tag_part(self, user_text_input, item_text_input):
        """
        只处理文本的结构
        """
        user_text_feature = self.feature_extracter_from_texts()(user_text_input) #(None,1,embedding)?
        item_text_feature = self.feature_extracter_from_texts()(item_text_input)
        x = Concatenate (name='concatenate_1')([user_text_feature, item_text_feature])
        for unit_num in self.content_fc_unit_nums: # 整合text
            x = Dense(unit_num, activation='relu')(x)
        return x

    def get_model(self):
        if self.model is None:
            user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                    name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
            item_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')

            predict_vector = self.get_text_tag_part(user_text_input, item_text_input)
            # 只处理文本时 merge feature后已经全连接，所以不要predict——fc
            predict_vector = Dropout(0.5)(predict_vector)
            predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)
            self.model = Model(inputs=[user_text_input, item_text_input],outputs=[predict_result])

            print('built whole model, done!')
            for layer in self.model.layers:
                print(layer.name)
            print ('built whole model, done!')
        return self.model

    def get_instances(self,mashup_id_instances, api_id_instances):
        """
        """
        examples=(
        np.array(self.encoded_texts.get_texts_in_index(mashup_id_instances, 'keras_setting', 0)),
        np.array(self.encoded_texts.get_texts_in_index(api_id_instances, 'keras_setting', self.num_users)),
        )

        return examples

    def get_mashup_api_text_features(self, mashup_ids = None, api_ids = None, save_results=True):
        """
        传入待测mashup的id列表，返回特征提取器提取的mashup的text的feature
        :param mashup_ids: 可以分别传入train和test的mashup
        :return: 默认输出mashup和api的text特征
        """

        mashup_text_middle_model = Model (inputs=[self.model.inputs[0]],
                                       outputs=[self.model.get_layer ('concatenate_1').input[0]])

        api_text_middle_model = Model (inputs=[self.model.inputs[1]],
                                       outputs=[self.model.get_layer ('concatenate_1').input[1]])

        if mashup_ids is None:
            mashup_ids=[i for i in range(self.num_users)]
        if api_ids is None:
            api_ids = [i for i in range(self.num_items)]

        mashup_instances_tuple = np.array(self.encoded_texts.get_texts_in_index(mashup_ids, 'keras_setting', 0)) # (np.array(mashup_ids))
        api_instances_tuple = np.array(self.encoded_texts.get_texts_in_index(api_ids, 'keras_setting', self.num_users)) # (np.array(api_ids))
        print('mashup_instances_tuple shape:', mashup_instances_tuple.shape)

        mashup_text_features=mashup_text_middle_model.predict ([mashup_instances_tuple], verbose=0) # 需要加0的索引？
        print('mashup_text_features shape:',mashup_text_features.shape)
        api_text_features = api_text_middle_model.predict ([api_instances_tuple], verbose=0)

        if save_results:
            mashup_text_features_path = os.path.join(dataset.crt_ds.model_path.format(self.get_simple_name()), 'mashup_text_features.dat')
            api_text_features_path = os.path.join(dataset.crt_ds.model_path.format(self.get_simple_name()),'api_test_features.dat')
            # save_2D_list(mashup_text_features, mashup_text_features_path, 'wb+')
            # save_2D_list(api_text_features, api_text_features_path, 'wb+')
            with open(mashup_text_features_path, 'ab+') as f:
                pickle.dump(mashup_text_features, f)
            with open(api_text_features_path, 'ab+') as f:
                pickle.dump(api_text_features, f)
        return mashup_text_features,api_text_features

# 待改！！！
class gx_text_only_MF_model(gx_text_only_model):
    """
    只处理text+MF
    """
    def __init__(self, base_dir, remove_punctuation, embedding_name, embedding_dim, text_extracter_mode,inception_channels, inception_pooling,
                 inception_fc_unit_nums, content_fc_unit_nums, mf_embedding_dim, mf_fc_unit_nums, predict_fc_unit_nums):
        super(gx_text_only_MF_model, self).__init__(base_dir, remove_punctuation, embedding_name, embedding_dim,text_extracter_mode,
                                                    inception_channels, inception_pooling, inception_fc_unit_nums,
                                                    content_fc_unit_nums, mf_embedding_dim, mf_fc_unit_nums)
        self.predict_fc_unit_nums=predict_fc_unit_nums # 用于整合文本和mf之后的预测

    def get_name(self):
        name = super(gx_text_only_model, self).get_name()
        name += 'predict_fc_unit_nums:{} '.format(self.predict_fc_unit_nums).replace(',', ' ');
        return 'gx_text_only_MF_model:' + name  # *** 用于区别每个模型  应包含选用的embedding，是否使用tag，inception结构，MF结构，总体结构（FC nums）

    def get_model(self):
        user_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',
                                name='user_text_input')  # MAX_SEQUENCE_LENGTH?  一个行向量
        item_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='item_text_input')
        x = self.get_text_tag_part(user_text_input, item_text_input)
        # x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        # print(x.shape) # (?, 1, 50)

        # left part
        user_id = Input(shape=(1,), dtype='int32', name='user_input') # 一个数字
        item_id = Input(shape=(1,), dtype='int32', name='item_input')
        # y = self.get_mf_part(user_id, item_id)

        # MLP part  使用接口model
        mf_mlp = self.get_mf_MLP(self.num_users, self.num_items, self.mf_embedding_dim, self.mf_fc_unit_nums)
        y= mf_mlp([user_id,item_id])

        #print(y.shape) # (?, 1, 50)

        # merge the two parts
        predict_vector = concatenate([x, y])
        print('final merge,done!')
        print(predict_vector.shape) # (?, 1, 100)

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense(unit_num, activation='relu')(predict_vector)

        # predict_vector=Flatten()(predict_vector)
        predict_vector = Dropout(0.5)(predict_vector)
        predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
            predict_vector)
        model = Model(inputs=[user_id,item_id,user_text_input, item_text_input], outputs=[predict_result])

        print('built whole model, done!')
        return model

    def get_instances(self, mashup_id_instances, api_id_instances):
        examples=(
        np.array(mashup_id_instances),
        np.array(api_id_instances),
        np.array(self.encoded_texts.get_texts_in_index(mashup_id_instances, 'keras_setting', 0)),
        np.array(self.encoded_texts.get_texts_in_index(api_id_instances, 'keras_setting', self.num_users)),
        )

        return examples


class gx_text_only_MLP_model(gx_text_only_model):
    def __init__(self,sim_mode):
        super (gx_text_only_MLP_model, self).__init__ ()

        self.sim_mode = sim_mode # mashup相似度计算方式

        self.num_feat = new_Para.param.num_feat # cf部分维度
        self.text_feature_dim = new_Para.param.inception_fc_unit_nums[-1]
        # i_factors_matrix 按照全局id排序; UI矩阵也是按照api id排序
        self.a_cf_features = np.zeros((meta_data.api_num,self.num_feat))

        print('a_id2index, size:', len(dataset.UV_obj.a_id2index))
        print('a_embeddings, shape:', dataset.UV_obj.a_embeddings.shape)
        for id,index in dataset.UV_obj.a_id2index.items():
            self.a_cf_features[id]=dataset.UV_obj.a_embeddings[index]

        # 而mashup不同，按照训练集中的内部索引大小
        self.his_mashup_ids=dataset.crt_ds.his_mashup_ids
        self.m_index2id = {index: id for id, index in dataset.UV_obj.m_id2index.items ()}
        self.m_id2index = dataset.UV_obj.m_id2index
        self.m_factors_matrix = dataset.UV_obj.m_embeddings

        self.model = None
        self.predict_fc_unit_nums = new_Para.param.predict_fc_unit_nums

        self.topK = new_Para.param.topK # 对模型有影响

        self.if_implict = new_Para.param.if_implict # 隐式
        self.CF_self_1st_merge= new_Para.param.CF_self_1st_merge
        self.cf_unit_nums=new_Para.param.cf_unit_nums

        # 显式共现
        self.if_explict = new_Para.param.if_explict # 0是没有，1是跟所有api的共现次数向量；2是跟最近邻mashup调用过的api的共现次数；3是最近邻mashup是否调用过该api，50D
        self.co_unit_nums=new_Para.param.shadow_co_fc_unit_nums

        self.m_feature_cosin_sim = -1*np.ones((self.num_users,self.num_users))

        self.m_cf_features = [] # 每个mashup对应的cf feature,按照全局id
        self.mid2neighIds={} # 每个mashup对应的近邻mashup id
        self.mashup_api_pair = meta_data.pd.get_mashup_api_pair('dict')  # 每个mashup调用的api序列

        ex_ = '_explict{}:{}'.format(self.if_explict, self.co_unit_nums).replace(',', ' ') if self.if_explict else ''
        im_ = '_implict:{}'.format(self.cf_unit_nums).replace(',', ' ') if self.if_implict else ''
        self.model_name += '_KNN_' + str(self.topK) + ex_ + im_

        self.simple_name = 'text_only_MLP'
        if self.if_implict:
            self.simple_name += '_implict'
        if self.if_explict:
            self.simple_name += '_explict'

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        return self.simple_name + self.model_name

    def cpt_mashup_sim(self,m_id1,m_id2,mode='feature_cosine'):
        if m_id1==m_id2:
            return 0
        if mode=='feature_cosine':
            if  self.m_feature_cosin_sim[m_id1][m_id2]<0:
                sim = cos_sim(self.m_text_features[m_id1],self.m_text_features[m_id2])
                self.m_feature_cosin_sim[m_id1][m_id2] = sim
                self.m_feature_cosin_sim[m_id2][m_id1] = sim
            else:
                sim = self.m_feature_cosin_sim[m_id1][m_id2]
            return sim

    def prepare(self,gx_text_only_model):
        self.m_text_features, self.a_text_features = gx_text_only_model.get_mashup_api_text_features(save_results=True) # 得到id到文本特征的映射

        # 得到每个mashup最近邻id，最近邻的cf-feature
        for m_id in range(self.num_users): # 每个mashup id
            index2sim=[(index,self.cpt_mashup_sim(m_id,self.his_mashup_ids[index],self.sim_mode))for index in range(len(self.his_mashup_ids))]
            max_k_pairs = heapq.nlargest(self.topK, index2sim, key=lambda x: x[1])  # 根据sim选取topK
            max_k_indexes, max_sims = zip(*max_k_pairs)
            self.mid2neighIds[m_id] = [self.his_mashup_ids[index] for index in max_k_indexes] # 局部索引-> id

            sims_array = np.array(max_sims) # 相似度归一化
            sims_array = sims_array/np.sum(sims_array)

            final_sims_array = np.zeros((len(self.his_mashup_ids),1)) # x,1
            for i in range(self.topK):
                final_sims_array[max_k_indexes[i]][0] = sims_array[i]
            m_feature = np.sum(final_sims_array*self.m_factors_matrix,axis=0)
            self.m_cf_features.append(m_feature)

        root = dataset.crt_ds.model_path.format(self.get_simple_name())
        if not os.path.exists(root):
            os.makedirs(root)
        m_feature_cosin_sim_path = os.path.join(root,'m_feature_cosin_sim')
        np.save(m_feature_cosin_sim_path,self.m_feature_cosin_sim)
        print('save m_feature_cosin_sim,done!')

    def get_model(self,_model):
        # 搭建简单模型
        if self.model is None:
            user_text_input = Input(shape=(self.text_feature_dim,), dtype='float32',name='user_text_input')
            item_text_input = Input(shape=(self.text_feature_dim,), dtype='float32', name='item_text_input')

            func_features_input = Concatenate()([user_text_input,item_text_input])
            predict_vector = func_features_input
            for index,unit_num in enumerate(self.content_fc_unit_nums):
                predict_vector = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg),name ='content_MLP_{}'.format(index)) (predict_vector)

            if self.if_implict: # 隐式历史交互
                mashup_cf_features = Input(shape=(self.num_feat,), dtype='float32')
                api_cf_features = Input(shape=(self.num_feat,), dtype='float32')
                if self.CF_self_1st_merge:
                    predict_vector2 = Concatenate() ([mashup_cf_features, api_cf_features])
                    for unit_num in self.cf_unit_nums:
                        predict_vector2 = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector2)
                    predict_vector=Concatenate()([predict_vector,predict_vector2])

            if self.if_explict:  # 显式历史交互
                co_vecs = Input(shape=(self.topK,), dtype='float32')
                predict_vector3 = Dense (self.co_unit_nums[0], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (co_vecs)
                for unit_num in self.co_unit_nums[1:]:
                    predict_vector3 = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector3)
                predict_vector = Concatenate()([predict_vector, predict_vector3])

            if self.if_implict or self.if_explict: # 只有CI时不需要
                for unit_num in self.predict_fc_unit_nums:
                    predict_vector = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector)

            predict_vector = Dropout (0.5) (predict_vector)
            predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
                predict_vector)
            if not self.if_implict and not self.if_explict:
                print('please use text_only model!')
                sys.exit()
            if self.if_implict and not self.if_explict: # 隐式
                self.model = Model(inputs=[user_text_input, item_text_input, mashup_cf_features, api_cf_features],
                                   outputs=[predict_result])
            if not self.if_implict and self.if_explict: # 显式
                self.model = Model(inputs=[user_text_input, item_text_input, co_vecs],
                                   outputs=[predict_result])
            if self.if_implict and  self.if_explict: # 显式+隐式
                self.model = Model(inputs=[user_text_input, item_text_input, mashup_cf_features, api_cf_features, co_vecs],
                                   outputs=[predict_result])
            print(self.get_name())
            print('build model,done!')

            w_dense2 = _model.get_layer('dense_2').get_weights()
            w_dense3 = _model.get_layer('dense_3').get_weights()
            w_dense4 = _model.get_layer('dense_4').get_weights()
            ws = [w_dense2,w_dense3,w_dense4]
            for index in range(len(self.content_fc_unit_nums)):
                self.model.get_layer('content_MLP_{}'.format(index)).set_weights(ws[index])
        return self.model

    def get_instances(self,mashup_id_instances, api_id_instances):
        results = []
        m_text_feature_instances= np.array([self.m_text_features[m_id] for m_id in mashup_id_instances])
        a_text_feature_instances = np.array([self.a_text_features[a_id] for a_id in api_id_instances])
        results.append(m_text_feature_instances)
        results.append(a_text_feature_instances)
        if self.if_implict:
            m_cf_feature_instances = np.array([self.m_cf_features[m_id] for m_id in mashup_id_instances])
            a_cf_feature_instances = np.array([self.a_cf_features[a_id] for a_id in api_id_instances])
            results.append(m_cf_feature_instances)
            results.append(a_cf_feature_instances)
        if self.if_explict:
            api_co_vecs = []
            for i in range(len(mashup_id_instances)):
                mashup_id = mashup_id_instances[i]
                api_id = api_id_instances[i]
                api_co_vec = [1.0 if api_id in self.mashup_api_pair[m_neigh_id] else 0 for m_neigh_id in self.mid2neighIds[mashup_id]]
                api_co_vecs.append(api_co_vec)
            results.append(np.array(api_co_vecs))

        return tuple(results)


