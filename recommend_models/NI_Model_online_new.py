import os
import pickle
import sys
sys.path.append("..")

from recommend_models.sequence import AttentionSequencePoolingLayer, SequencePoolingLayer, DNN
from recommend_models.utils import NoMask
from recommend_models.CI_Model import CI_Model
from main.new_para_setting import new_Para
from main.dataset import meta_data, dataset
from Helpers.util import cos_sim, save_2D_list

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, Input, concatenate, Concatenate, Embedding, Multiply, PReLU, AveragePooling1D, \
    BatchNormalization, AveragePooling2D,Dropout, Lambda, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant


# 针对新场景，在内容交互的基础上搭建新的完整模型:
# 可以完全依赖online_node2vec得到新mashup的隐式表示
# NI部分只使用tag feature+ cosine作为mashup相似度，暂不考虑已选择服务，换句话说，为每个mashup固定近邻和node2vec空间的特征表示

class NI_Model_online(CI_Model):
    def __init__(self, old_new, if_implict,if_explict,if_correlation,eachPath_topK=True):
        super(NI_Model_online, self).__init__(old_new)

        # i_factors_matrix 按照全局id排序; UI矩阵也是按照api id从小到大排序
        # 而mashup不同，按照训练集中的内部索引大小
        self.num_feat = new_Para.param.num_feat
        self.topK = new_Para.param.topK
        self.m_id2index = dataset.UV_obj.m_id2index
        self.m_index2id = {index: id for id, index in dataset.UV_obj.m_id2index.items()}

        self.NI_handle_slt_apis_mode = new_Para.param.NI_handle_slt_apis_mode
        self.if_implict = if_implict
        self.CF_self_1st_merge = new_Para.param.CF_self_1st_merge
        self.cf_unit_nums = new_Para.param.cf_unit_nums

        self.if_explict = if_explict  # 0是没有，1是跟所有api的共现次数向量；2是跟最近邻mashup调用过的api的共现次数；3是最近邻mashup是否调用过该api，50D
        self.co_unit_nums = new_Para.param.shadow_co_fc_unit_nums

        # 可组合性
        self.if_correlation = if_correlation
        self.cor_fc_unit_nums = new_Para.param.cor_fc_unit_nums

        self.predict_fc_unit_nums = new_Para.param.predict_fc_unit_nums
        self.NI_OL_mode = new_Para.param.NI_OL_mode
        self.lr = new_Para.param.NI_learning_rate
        self.optimizer = Adam(self.lr)

        self.model = None

        self.eachPath_topK = eachPath_topK

        self.mid2NI_sims = None # 每个mashup到历史mashup的简单相似度
        self.mid2NI_feas = {}
        self.maid2NI_feas = {}
        self.mid_sltAids_2NI_feas = {}
        self.mid2neighors = {}

        # set_simple_name
        self.name = ''
        if self.if_explict :
            self.name += 'explict'
        if self.if_implict:
            self.name += 'implict'
        if self.if_correlation:
            self.name += 'cor'
        self.name += str(self.topK) # 近邻规模

        self.simple_name = 'new_{}_OL_' if self.old_new == 'new' else 'old_{}_OL_'
        self.simple_name = self.simple_name.format(self.name)

        if self.if_implict:  # 隐式交互
            if self.old_new == 'new':
                if not self.NI_handle_slt_apis_mode:
                    self.simple_name += 'noSlts_'
                else:
                    self.simple_name += (self.NI_handle_slt_apis_mode[:3] + '_')

            self.simple_name += (new_Para.param.mf_mode+'_')
            if not self.CF_self_1st_merge:
                self.simple_name += 'NoMLP_'

        self.simple_name += ('_{}'.format(self.lr))
        # self.simple_name += '_reductData' # !!!

        # set_model_name
        self.eachPath_topK_name = 'eachPathTopK' if self.eachPath_topK else 'allPathsTopK'
        ex_ = '_explict{}:{}'.format(self.if_explict, self.co_unit_nums).replace(',', ' ') if self.if_explict else ''
        im_ = '_implict:{}'.format(self.cf_unit_nums).replace(',', ' ') if self.if_implict else ''
        correlation = '_correlation:{}'.format(self.cor_fc_unit_nums).replace(',', ' ') if self.if_correlation else ''
        self.model_name +=  self.simple_name+ ex_ + im_ + correlation

        # set_paths 2
        self.model_dir = os.path.join(dataset.crt_ds.model_path, '{}_{}').format(self.simple_name,self.NI_OL_mode,self.eachPath_topK_name) # _1head_3
        self.NI_features_path = os.path.join(self.model_dir, 'NI_features.fea')
        self.train_slt_apis_mid_features_path = os.path.join(self.model_dir,'train_slt_apis_mid_features.csv')
        self.test_slt_apis_mid_features_path = os.path.join(self.model_dir, 'test_slt_apis_mid_features.csv')

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        """
        self.model_name区分模型框架，返回的name用于记录在evalute中，区分不同的模型，架构
        :return:
        """
        return self.model_name

    def set_mashup_api_features(self, recommend_model):
        """
        设置mashup和api的text和tag特征，用于计算相似度，进而计算mashup的NI表示;
        在get_model()和get_instances()之前设置
        :param recommend_model: 利用CI模型获得所有特征向量
        :return:
        """
        self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features = \
            recommend_model.get_mashup_api_features(self.all_mashup_num, self.all_api_num)
        # api 需要增加一个全为0的，放在最后，id为api_num，用来对slt_apis填充
        self.api_tag_features = np.vstack((self.api_tag_features, np.zeros((1, self.word_embedding_dim))))
        self.api_texts_features = np.vstack((self.api_texts_features, np.zeros((1, self.inception_fc_unit_nums[-1]))))
        self.features = (self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features)
        self.CI_path = recommend_model.model_dir

    def set_attribute(self):
        self.all_mashup_num = meta_data.mashup_num
        self.all_api_num = meta_data.api_num
        self.his_m_ids = dataset.crt_ds.his_mashup_ids
        self.his_m_ids_set = set(self.his_m_ids)
        self.mashup_api_pairs = meta_data.pd.get_mashup_api_pair('dict')  #
        self.train_mashup_api_dict = {key: value for key, value in self.mashup_api_pairs.items() if key in self.his_m_ids_set}

    def prepare_sims(self, sim_model=None, train_data=None, test_data=None):
        # 准备各种相似度：CI_recommend_model可以是提供文本tag特征的CI，也可以是提供相似度支持的HINRec_model
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.set_attribute()

        self.his_mashup_NI_feas = np.array([dataset.UV_obj.m_embeddings[self.m_id2index[his_m_id]] for his_m_id in self.his_m_ids])
        if self.NI_OL_mode == 'tagSim': # 基于CI部门的特征计算相似度，MISR使用,待修改!!!
            self.set_mashup_api_features(sim_model)
        else:
            self.mid2neighors_path = os.path.join(sim_model.model_dir, 'mid2neighors.dat')
            self.path_weights = sim_model.path_weights # 读取预训练的相似度模型中的meta-path权重

            self.mID2AllSimsPath = os.path.join(sim_model.model_dir, 'mID2AllSims_{}.sim'.format(self.NI_OL_mode))
            self.mID2ASimPath = os.path.join(sim_model.model_dir, 'mID2ASim_{}_{}_{}.sim'.format(self.NI_OL_mode,self.eachPath_topK_name,self.topK))
            self.mID2ASim,self.mID2AllSims = {},{}

            all_paths_sim_modes = ['PasRec','PasRec_2path','IsRec','IsRec_best']
            if self.NI_OL_mode in all_paths_sim_modes: # 计算mashup表示时需要已选择服务()
                self.mid_sltAids_2NI_feas_path = os.path.join(sim_model.model_dir, 'NI_OL_m_id2{}_{}_{}.feas'.format(self.NI_OL_mode,self.eachPath_topK_name,self.topK))
                self.get_samples_m_feas(train_data, test_data,sim_model)

    def get_mid2AllSims(self, train_data, test_data, sim_model):
        # 读取mid/(mid,slt_aids)到每个历史mashup的相似度（IsRec的已经剪枝，只有一部分mashup）
        if os.path.exists(self.mID2AllSimsPath):  # !!! 为了显示不得已，再改  self.mid2neighors[key]
            with open(self.mID2AllSimsPath, 'rb') as f:
                self.mID2AllSims = pickle.load(f)
        else:
            # 先求self.mID2AllSims，后存
            if len(train_data[:-1]) == 3:
                mashup_id_list, api_id_list, mashup_slt_apis_list = train_data[:-1]
            else:
                mashup_id_list, api_id_list = train_data[:-1]
            for i in range(len(mashup_id_list)):
                if self.NI_OL_mode == 'PasRec' or self.NI_OL_mode == 'IsRec': # 需要已选择的服务
                        key = (mashup_id_list[i], tuple(mashup_slt_apis_list[i]))
                        if key not in self.mID2AllSims.keys():
                            self.mID2AllSims[key] = sim_model.get_id2PathSims(*key,if_temp_save=False)
                else:
                    key = mashup_id_list[i]
                    if key not in self.mID2AllSims.keys():
                        self.mID2AllSims[key] = sim_model.get_id2PathSims(key,if_temp_save=False)
            print('compute id2AllPathSims for train_data,done!')

            if len(test_data[:-1]) == 3:
                mashup_id_list, api_id_list, mashup_slt_apis_list = test_data[:-1]
            else:
                mashup_id_list, api_id_list = test_data[:-1]
            for i in range(len(mashup_id_list)):
                for j in range(len(api_id_list[i])):
                    if self.NI_OL_mode == 'PasRec' or self.NI_OL_mode == 'IsRec':  # 需要已选择的服务
                        key = (mashup_id_list[i][j], tuple(mashup_slt_apis_list[i]))
                        if key not in self.mID2AllSims.keys():
                            self.mID2AllSims[key] = sim_model.get_id2PathSims(*key, if_temp_save=False)
                    else:
                        key = mashup_id_list[i][j]
                        if key not in self.mID2AllSims.keys():
                            self.mID2AllSims[key] = sim_model.get_id2PathSims(key, if_temp_save=False)
            print('compute id2AllPathSims for test_data,done!')

            # with open(self.mID2AllSimsPath, 'wb') as f:
            #     pickle.dump(self.mID2AllSims, f)
        return self.mID2AllSims

    def get_mID2ASim(self, train_data, test_data, sim_model):
        """得到一个mashup到其他mashup的归一化的综合相似度向量"""
        if os.path.exists(self.mid2neighors_path) and os.path.exists(self.mID2ASimPath): # ...计算explicit用
            with open(self.mID2ASimPath, 'rb') as f:
                self.mID2ASim = pickle.load(f)
            with open(self.mid2neighors_path, 'rb') as f:
                self.mid2neighors = pickle.load(f)
        else: # 一次性计算全部的并存储
            print('mID2ASim not exist, computing!')
            dict_ = self.get_mid2AllSims(train_data, test_data, sim_model) # self.mID2AllSims 每个sample的相似度映射
            for key,id2PathSims in dict_.items():
                m_id = key if isinstance(key,int) else key[0] # mashup ID

                if self.eachPath_topK: # 每个路径的topK
                    for i in range(len(id2PathSims)): # 某一种路径的相似度
                        id2PathSim = id2PathSims[i]
                        num = min(self.topK, len(id2PathSim))
                        id2PathSim = sorted(id2PathSim.items(), key=lambda x: x[1], reverse=True)[:num]
                        id2PathSims[i] = {key:value for key,value in id2PathSim}

                id2score = {his_m_id: 0 for his_m_id in self.his_m_ids} # 到所有历史mashup的综合相似度
                for his_m_id in id2score.keys(): # 每个历史近邻mashup
                    if his_m_id != m_id:  # 除去自身
                        for path_index, id2aPathSim in enumerate(id2PathSims): # 每种相似度路径
                            pathSim = 0 if his_m_id not in id2aPathSim.keys() else id2aPathSim[his_m_id] # 某个历史mid可能没有某种相似度
                            id2score[his_m_id] += pathSim * self.path_weights[path_index]

                # 为显式设计，综合所有路径之后存储topk近邻
                num = min(self.topK, len(id2score))
                self.mid2neighors[key], _ = zip(*(sorted(id2score.items(), key=lambda x: x[1], reverse=True)[:num]))  # 按顺序存储topK个近邻的ID

                if not self.eachPath_topK: # 最终所有路径综合评分的topK
                    num = min(self.topK, len(id2score))
                    id2score = sorted(id2score.items(), key=lambda x: x[1], reverse=True)[:num]
                    id2score = {key: value for key, value in id2score}

                sims = np.array([id2score[his_m_id] if his_m_id in id2score.keys() else 0  for his_m_id in self.his_m_ids])  # 按顺序排好的sims: (#his_m_ids)
                sum_sim = sum(sims)
                if sum_sim == 0:
                    print('sims sum=0!')
                else:
                    sims = sims / sum_sim
                self.mID2ASim[key] = sims

            print('mID2ASim, computed!')
            with open(self.mID2ASimPath, 'wb') as f:
                pickle.dump(self.mID2ASim, f)

            with open(self.mid2neighors_path, 'wb') as f:
                pickle.dump(self.mid2neighors, f)
        return self.mID2ASim

    def get_samples_m_feas(self,train_data,test_data,sim_model):
        if os.path.exists(self.mid2neighors_path) and os.path.exists(self.mid_sltAids_2NI_feas_path): # 一步到位，读取特征
            with open(self.mid_sltAids_2NI_feas_path, 'rb') as f:
                self.mid_sltAids_2NI_feas = pickle.load(f)
            with open(self.mid2neighors_path, 'rb') as f:
                self.mid2neighors = pickle.load(f)
        else:
            print('mid_sltAids_2NI_feas not exist, computing!')
            dict_ = self.get_mID2ASim(train_data, test_data, sim_model) # 综合sim的映射 self.mID2ASim
            print('compute mid_sltAids_2NI_feas,done!')
            # 获得self.id2NI_sims之后即时计算全部样本的NI feature
            for key,value in dict_.items():
                self.mid_sltAids_2NI_feas[key] = np.dot(value,self.his_mashup_NI_feas)

            with open(self.mid_sltAids_2NI_feas_path, 'wb') as f:
                pickle.dump(self.mid_sltAids_2NI_feas, f)

    def set_embedding_matrixs(self):
        self.i_factors_matrix = np.zeros((meta_data.api_num + 1, self.num_feat))
        a_embs_array = np.array(dataset.UV_obj.a_embeddings)
        for id, index in dataset.UV_obj.a_id2index.items():
            self.i_factors_matrix[id] = a_embs_array[index]

    def set_embedding_layers(self):
        self.api_implict_emb_layer = Embedding(self.all_api_num + 1,
                                               new_Para.param.num_feat,
                                               embeddings_initializer=Constant(self.i_factors_matrix),
                                               mask_zero=False,
                                               trainable=False,
                                               name='api_implict_embedding_layer')

    def prepare(self):
        self.set_embedding_matrixs()
        self.set_embedding_layers()

    def get_model(self):
        if not self.model:
            mashup_fea_input = Input(shape=(self.num_feat,), dtype='float32', name='NI_mashup_fea_input')  # (None,25)
            api_id_input = Input(shape=(1,), dtype='int32', name='NI_api_id_input')
            inputs = [mashup_fea_input, api_id_input]

            self.prepare()
            api_implict_embs = self.api_implict_emb_layer(api_id_input)  # (None,1,25)
            api_implict_embs_2D = Lambda(lambda x: tf.squeeze(x, axis=1))(api_implict_embs) # (None,25)
            feature_list = [mashup_fea_input,api_implict_embs_2D]

            if self.old_new == 'new' and self.NI_handle_slt_apis_mode:
                mashup_slt_apis_input = Input(shape=(new_Para.param.slt_item_num,), dtype='int32', name='slt_apis_input')
                inputs.append(mashup_slt_apis_input)
                keys_slt_api_implict_embs = self.api_implict_emb_layer(mashup_slt_apis_input)  # (None,3,25)

                if self.NI_handle_slt_apis_mode in ('attention','average') :
                    mask = Lambda(lambda x: K.not_equal(x, self.all_api_num))(mashup_slt_apis_input)  # (?, 3) !!!
                    if self.NI_handle_slt_apis_mode == 'attention':
                        slt_api_implict_embs_hist = AttentionSequencePoolingLayer(supports_masking=True)([api_implict_embs, keys_slt_api_implict_embs],mask=mask)
                    else:  # 'average'
                        slt_api_implict_embs_hist = SequencePoolingLayer('mean', supports_masking=True)(keys_slt_api_implict_embs, mask=mask)
                    slt_api_implict_embs_hist = Lambda(lambda x: tf.squeeze(x, axis=1))(slt_api_implict_embs_hist)  # (?, 1, 25)->(?, 25)
                elif self.NI_handle_slt_apis_mode == 'full_concate':
                    slt_api_implict_embs_hist = Reshape((new_Para.param.slt_item_num*self.num_feat,))(keys_slt_api_implict_embs)  # (?,75)
                else:
                    raise ValueError('wrong NI_handle_slt_apis_mode!')
                feature_list.append(slt_api_implict_embs_hist)

            feature_list = list(map(NoMask(), feature_list))  # DNN不支持mak，所以不能再传递mask
            all_features = Concatenate(name='all_emb_concatenate')(feature_list)

            output = DNN(self.cf_unit_nums[:-1])(all_features)
            output = Dense(self.cf_unit_nums[-1], activation='relu', kernel_regularizer=l2(new_Para.param.l2_reg),name='implict_dense_{}'.format(len(self.cf_unit_nums)))(output)

            # 输出层
            if new_Para.param.final_activation == 'softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(output)
            elif new_Para.param.final_activation == 'sigmoid':
                predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(output)

            self.model = Model(inputs=inputs, outputs=[predict_result], name='predict_model')

            for layer in self.model.layers:
                print(layer.name)
            if not os.path.exists(self.model_name_path):
                with open(self.model_name_path, 'w+') as f:
                    f.write(self.get_name())
        return self.model

    def get_instances(self, mashup_id_list, api_id_list, mashup_slt_apis_list=None, test_phase_flag=True): # if_Train最后一个参数：pairwise型或者online场景下的区别训练测试时的mashup表示
        if self.if_explict: # 临时
            co_vecs = np.array([[1.0 if api_id_list[i] in self.train_mashup_api_dict[neighbor_id] else 0.0 for neighbor_id in self.mid2neighors[mashup_id_list[i]]] for i in range(len(mashup_id_list))])
            return co_vecs

        if self.NI_OL_mode == 'PasRec' or self.NI_OL_mode == 'IsRec':
            mashup_fea_list = [self.mid_sltAids_2NI_feas[(mashup_id_list[i], tuple(mashup_slt_apis_list[i]))] for i in range(len(mashup_id_list))]
            mashup_fea_array = np.array(mashup_fea_list, dtype='float32')
        if self.NI_OL_mode == 'PasRec_2path' or self.NI_OL_mode == 'IsRec_best':
            mashup_fea_list = [self.mid_sltAids_2NI_feas[m_id] for m_id in mashup_id_list]
            mashup_fea_array = np.array(mashup_fea_list)

        if self.old_new == 'new' and self.NI_handle_slt_apis_mode:
            # padding, the index of slt_item_num be mapped to zero in the embedding layer
            mashup_slt_apis_array = np.ones((len(mashup_id_list), new_Para.param.slt_item_num),dtype='int32') * self.all_api_num  # default setting,api_num
            if self.old_new == 'new':
                for i in range(len(mashup_slt_apis_list)):
                    for j in range(len(mashup_slt_apis_list[i])):
                        mashup_slt_apis_array[i][j] = mashup_slt_apis_list[i][j]
            results = (mashup_fea_array, np.array(api_id_list),mashup_slt_apis_array)
        else:
            results = (mashup_fea_array, np.array(api_id_list))
        return results