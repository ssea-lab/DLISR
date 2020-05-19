import os
import sys

sys.path.append("..")
from recommend_models.midMLP_model import midMLP_feature_obj
from recommend_models.attention_block import MultiHeadAttention, new_attention_3d_block

import tensorflow as tf
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Concatenate, Dense, Dropout, Lambda, AveragePooling2D, Reshape
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.regularizers import l2
from main.new_para_setting import new_Para
from recommend_models.CI_Model import CI_Model
from recommend_models.NI_Model_online_new import NI_Model_online
from recommend_models.recommend_Model import gx_model
import numpy as np


class top_MLP(object):
    def __init__(self,CI_recommend_model= None,CI_model=None,CI_model_dir=None,NI_recommend_model=None,NI_model=None,NI_recommend_model2=None,NI_model2=None,new_old='new'):
        # NI_recommend_model用于隐式交互，NI_recommend_model2用于显示交互
        if CI_recommend_model is None and CI_model_dir is None:
            raise ValueError('CI_recommend_model and CI_model_dir cannot be None at the same time!')

        # 路径在CI下面，依次是CI/top_MLP
        if CI_model_dir is None and CI_recommend_model is not None:
            self.CI_model_dir = CI_recommend_model.model_dir

        self.new_old = new_old
        self.CI_dim = new_Para.param.content_fc_unit_nums[-1]
        self.NI_dim = new_Para.param.cf_unit_nums[-1]
        self.NI_dim2 = new_Para.param.cor_fc_unit_nums[-1] if self.new_old=='new' else new_Para.param.shadow_co_fc_unit_nums[-1]
        self.predict_fc_unit_nums = new_Para.param.predict_fc_unit_nums

        self.model = None
        self.CI_recommend_model = CI_recommend_model
        self.NI_recommend_model = NI_recommend_model
        self.NI_recommend_model2 = NI_recommend_model2
        self.CI_model = CI_model
        self.NI_model = NI_model
        self.NI_model2 = NI_model2

        if self.CI_recommend_model is not None and self.CI_model is not None:
            self.CI_midMLP_feature_obj = midMLP_feature_obj(self.CI_recommend_model, self.CI_model)
        else:
            self.CI_midMLP_feature_obj =  None

        if self.NI_recommend_model is not None and self.NI_model is not None:
            self.NI_midMLP_feature_obj = midMLP_feature_obj(self.NI_recommend_model, self.NI_model)
        else:
            self.NI_midMLP_feature_obj =  None

        if self.NI_recommend_model2 is not None and self.NI_model2 is not None:
            self.NI_midMLP_feature_obj2 = midMLP_feature_obj(self.NI_recommend_model2, self.NI_model2,if_explict= True)
        else:
            self.NI_midMLP_feature_obj2 =  None

        self.lr = new_Para.param.topMLP_learning_rate
        self.optimizer = Adam(lr=self.lr)

        self.set_simple_name()
        self.set_model_name()
        self.set_paths()


    def set_simple_name(self):
        CI_name = 'CI' if self.CI_recommend_model is not None else ''
        NI_name1 = 'implict' if self.NI_recommend_model is not None else ''
        if self.NI_recommend_model2 is None:
            NI_name2 = ''
        else:
            if self.new_old=='new':
                NI_name2 = 'cor'
            elif self.new_old=='old':
                NI_name2 = 'explict'

        self.simple_name = '{}_{}_{}_{}_{}_{}_topMLP{}{}top{}'.format(CI_name,new_Para.param.simple_CI_mode, NI_name1,new_Para.param.simple_CI_mode,new_Para.param.NI_OL_mode,NI_name2,
                                                                      self.predict_fc_unit_nums, self.lr,
                                                                      new_Para.param.topK).replace(',', '_')

    def get_simple_name(self):
        return self.simple_name

    def set_model_name(self):
        self.model_name = self.simple_name+'_{}'.format(self.predict_fc_unit_nums).replace(',', ' ')

    def set_paths(self):
        self.model_dir = os.path.join(self.CI_model_dir,self.simple_name) # 放在CI路径下，可能使用显式和隐式
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')

    def get_name(self):
        """
        self.model_name区分模型框架，返回的name用于记录在evalute中，区分不同的模型，架构
        :return:
        """
        return self.model_name

    def get_model(self):
        if self.model is None:
            CI_input = Input(shape=(self.CI_dim,), dtype='float32', name='CI_input')
            NI_input = Input(shape=(self.NI_dim,), dtype='float32', name='implict_NI_input')
            NI_input2 = Input(shape=(self.NI_dim2,), dtype='float32', name='explict_NI_input')
            if self.CI_model is None: # INI+ENI
                inputs = [NI_input, NI_input2]
            else:
                if self.NI_model is None and self.NI_model2 is not None:
                    inputs = [CI_input, NI_input2]
                if self.NI_model2 is None and self.NI_model is not None:
                    inputs = [CI_input, NI_input]
                if self.NI_model is not None and self.NI_model2 is not None:
                    inputs = [CI_input, NI_input, NI_input2]
            predict_vector = Concatenate()(inputs)

            for index, unit_num in enumerate(self.predict_fc_unit_nums):
                predict_vector = Dense(unit_num, name='predict_dense_{}'.format(index), activation='relu',kernel_regularizer=l2(new_Para.param.l2_reg))(predict_vector)
            predict_vector = Dropout(0.5)(predict_vector)

            if new_Para.param.final_activation == 'softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(predict_vector)
            elif new_Para.param.final_activation == 'sigmoid':
                predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)
            print('build top_MLP model,done!')

            self.model = Model(inputs=inputs,outputs=[predict_result],name='top_MLP')
            if not os.path.exists(self.model_name_path):
                with open(self.model_name_path, 'w+') as f:
                    f.write(self.get_name())
        return self.model

    # 利用CI和NI的中间交互特征，训练上层MLP
    def get_instances(self, mashup_id_list, api_id_list, mashup_slt_apis_list=None,test_phase_flag=True):
        # 最后一个参数供pairwise和训练测试不同的NI使用, 现在暂时只有隐式近邻有用
        if self.CI_model is not None:
            CI_midMLP_features = self.CI_midMLP_feature_obj.get_midMLP_feature('text_tag_feature_extracter',mashup_id_list,
                                                                               api_id_list, mashup_slt_apis_list)

        if self.NI_model is not None:
            NI_layer_name = 'implict_dense_{}'.format(len(new_Para.param.cf_unit_nums)) # 中间层名字
            NI_midMLP_features = self.NI_midMLP_feature_obj.get_midMLP_feature(NI_layer_name, mashup_id_list,
                                                                               api_id_list, mashup_slt_apis_list,test_phase_flag=test_phase_flag)
        if self.NI_model2 is not None:
            name = 'explict' if self.new_old =='old' else 'cor'
            NI_layer_name2 = '{}_dense_{}'.format(name,len(new_Para.param.shadow_co_fc_unit_nums)) # 中间层名字
            NI_midMLP_features2 = self.NI_midMLP_feature_obj2.get_midMLP_feature(NI_layer_name2, mashup_id_list,
                                                                               api_id_list, mashup_slt_apis_list)
        # 中间结果作为输入
        if self.CI_model is None:  # INI+ENI
            return (np.array(NI_midMLP_features),np.array(NI_midMLP_features2))
        else:
            if self.NI_model is None and self.NI_model2 is not None:
                return (np.array(CI_midMLP_features),np.array(NI_midMLP_features2))
            if self.NI_model2 is None and self.NI_model is not None:
                return (np.array(CI_midMLP_features),np.array(NI_midMLP_features))
            if self.NI_model is not None and self.NI_model2 is not None:
                return (np.array(CI_midMLP_features),np.array(NI_midMLP_features),np.array(NI_midMLP_features2))

    def save_sth(self):
        if self.CI_midMLP_feature_obj is not None:
            self.CI_midMLP_feature_obj.save_sth()
        if self.NI_midMLP_feature_obj is not None:
            self.NI_midMLP_feature_obj.save_sth()
        if self.NI_midMLP_feature_obj2 is not None:
            self.NI_midMLP_feature_obj2.save_sth()


# 也可以根据两个训练好的模型CI和NI，集成学习

# fine-tuning,可以只调优上层MLP，也可以改变CI和NI中的attention
class fine_Tune(object):
    def __init__(self, CI_recommend_model,CI_model, NI_recommend_model,  NI_model, top_MLP_recommend_model=None,top_MLP_model=None, model_mode='ft',ft_mode='',lr=0.0001):
        """
        # 传入一个训练好的CI,NI,top_MLP_model模型
        :param CI_model:
        :param NI_model:
        :param top_MLP_model: fine_tune时利用其参数初始化
        :param model_mode : 是基于已有模型再finetune还是参数完全随机化，'co_train'  'ft'
        :param ft_mode：topMLP;3MLP(也会更新特征交互时的attention); whole
        """
        self.model = None
        self.model_mode = model_mode
        self.CI_recommend_model = CI_recommend_model
        self.NI_recommend_model = NI_recommend_model
        self.top_MLP_recommend_model = top_MLP_recommend_model
        self.CI_model = CI_model
        self.NI_model = NI_model
        self.top_MLP_model = top_MLP_model # 如果是fine-tune，必须非None

        self.lr = lr # 起初0.0001 0.0003  学习率特别小
        self.optimizer = SGD(lr=self.lr)
        self.predict_fc_unit_nums = new_Para.param.predict_fc_unit_nums

        self.ft_mode = ft_mode # 默认只训练topMLP,可以三个MLP连同attention一起训练，还是连同特征提取器一起训练
        if self.model_mode == 'co_train':
            self.simple_name = 'co_train_CI_{}_NI_{}_{}'.format(new_Para.param.simple_CI_mode,new_Para.param.NI_OL_mode,self.lr)
        elif self.model_mode == 'ft':
            self.simple_name = 'ft_{}_{}_{}-2'.format(ft_mode, self.top_MLP_recommend_model.simple_name,self.lr)
        self.model_name = self.simple_name  # 再改！！！
        # 路径在NI下面，一次是CI/NI/top_MLP
        # self.model_dir = os.path.join(top_MLP_recommend_model.model_dir,self.simple_name)
        self.model_dir = os.path.join(CI_recommend_model.model_dir, self.simple_name) # CI路径下
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')

    def get_model(self):
        if self.model is None:
            CI_feature = self.CI_model.get_layer('text_tag_feature_extracter').output

            NI_layer_name = 'implict_dense_{}'.format(len(new_Para.param.cf_unit_nums) - 1)  # 中间层名字
            NI_feature = self.NI_model.get_layer(NI_layer_name).output

            predict_vector = Concatenate()([CI_feature, NI_feature])
            for index, unit_num in enumerate(self.predict_fc_unit_nums):
                predict_vector = Dense(unit_num, name='predict_dense_{}'.format(index), activation='relu',
                                       kernel_regularizer=l2(new_Para.param.l2_reg))(predict_vector)
            predict_vector = Dropout(0.5)(predict_vector)

            if new_Para.param.final_activation == 'softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(predict_vector)
            elif new_Para.param.final_activation == 'sigmoid':
                predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
                    predict_vector)

            print('build whole fine-tuning model,done!')
            self.model = Model(inputs=[*self.CI_model.input, *self.NI_model.input], outputs=[predict_result], name='whole_model')
            for layer in self.model.layers:  # 打印所有名字
                print(layer.name)

            if not os.path.exists(self.model_name_path):
                with open(self.model_name_path, 'w+') as f:
                    f.write(self.get_name())
            return self.model

    def pre_fine_tune(self):
            layer_num_below_top_MLP = 2 + len(self.predict_fc_unit_nums) # MLP,dropout,prediction
            # 先将下层设为不可训练，上层MLP(predict_dense)可训练
            for layer in self.model.layers[:-1*layer_num_below_top_MLP]:
                layer.trainable = False
            for layer in self.model.layers[-1*layer_num_below_top_MLP:]:
                layer.trainable = True

            # top_MLP参数初始化
            for index in range(len(self.predict_fc_unit_nums)):
                layer_name = 'predict_dense_{}'.format(index)
                weights = self.top_MLP_model.get_layer(layer_name).get_weights()
                self.model.get_layer(layer_name).set_weights(weights)

            def train_CINI():
                # CI中MLP设为可训练
                for index in range(len(new_Para.param.content_fc_unit_nums[:-1])):
                    CI_MLP_layer_name = 'content_dense_{}'.format(index)
                    self.model.get_layer(CI_MLP_layer_name).trainable = True
                self.model.get_layer('text_tag_feature_extracter').trainable = True

                # NI中MLP设为可训练
                for index in range(len(new_Para.param.cf_unit_nums)):
                    NI_MLP_layer_name = 'implict_dense_{}'.format(index)
                    self.model.get_layer(NI_MLP_layer_name).trainable = True

                if 'attention' in new_Para.param.CI_handle_slt_apis_mode:
                    # CI,NI的特征提取部分不变, attention_block的训练
                    self.model.get_layer('MH_whole_MultiHeadAttBlock').trainable = True
                    self.model.get_layer('implict_MH_whole_MultiHeadAttBlock').trainable = True
                    # self.CI_model.get_layer('text_attBlock').trainable = True
                    # self.CI_model.get_layer('tag_attBlock').trainable = True
                    # self.NI_model.get_layer('implict_attBlock').trainable = True

            if self.ft_mode == 'topMLP':
                pass
            elif self.ft_mode == '3MLP': # MLP联合attention一起训练
                train_CINI()
            elif self.ft_mode == 'whole': # 彻底地联合训练，只针对OL_GE，因为基于CI的NI(tag_sim等)和CI联合训练效率很低
                # 不能这么写，因为model中很多embedding层(id2text/tag的encoding的embedding层)是不可训练的
                # for layer in self.model.layers[:-1 * layer_num_below_top_MLP]:
                #     layer.trainable = True

                # 实际上很多lambda层，reshape，concatenate层都没有参数，不用管；
                # 只需要注意CI的特征提取器(Model写法，内含词的embedding层，inception等)；CI和NI中特征交互的MLP,attention块等
                train_CINI()
                self.model.get_layer('text_feature_extracter').trainable = True
                self.model.get_layer('categories_feature_extracter').trainable = True

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        """
        self.model_name区分模型框架，返回的name用于记录在evalute中，区分不同的模型，架构
        :return:
        """
        return self.simple_name + self.model_name

    # 利用两个模型的输入
    def get_instances(self, mashup_id_list, api_id_list, mashup_slt_apis_list=None,test_phase_flag=True):
        CI_examples = self.CI_recommend_model.get_instances(mashup_id_list, api_id_list, mashup_slt_apis_list)
        NI_examples = self.NI_recommend_model.get_instances(mashup_id_list, api_id_list, mashup_slt_apis_list,test_phase_flag=test_phase_flag)
        return (*CI_examples,*NI_examples)


# <Software Service Recommendation Base on Collaborative Filtering Neural Network Model> #
class PNCF_doubleTower(object):
    # 双塔模型，将mashup和service的特征分别整合，然后使用MLP学习
    def __init__(self,CI_recommend_model= None,CI_model=None,CI_model_dir=None,NI_recommend_model=None,NI_model=None,new_old='new',model_name = 'PNCF'):
        # NI_recommend_model用于隐式交互，NI_recommend_model2用于显示交互
        if CI_recommend_model is None and CI_model_dir is None:
            raise ValueError('CI_recommend_model and CI_model_dir cannot be None at the same time!')

        # 路径在CI下面，依次是CI/top_MLP
        if CI_model_dir is None and CI_recommend_model is not None:
            CI_model_dir = CI_recommend_model.model_dir

        self.model_name = model_name
        self.new_old = new_old
        self.CI_dim = 50*2
        self.NI_dim = 25
        self.predict_fc_unit_nums = new_Para.param.predict_fc_unit_nums

        self.model = None
        self.CI_recommend_model = CI_recommend_model
        self.NI_recommend_model = NI_recommend_model
        self.CI_model = CI_model
        self.NI_model = NI_model

        self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features = \
            self.CI_recommend_model.get_mashup_api_features(self.CI_recommend_model.all_mashup_num, self.CI_recommend_model.all_api_num)
        self.mashup_CI_features = np.hstack((np.array(self.mashup_texts_features),np.array(self.mashup_tag_features)))
        self.api_CI_features = np.hstack((np.array(self.api_texts_features), np.array(self.api_tag_features)))

        self.lr = new_Para.param.CI_learning_rate # 暂时用CI的
        self.optimizer = Adam(lr=self.lr)

        self.simple_name = '{}_CI_{}_INI_{}_topMLP{}{}top{}_2'.format(self.model_name,new_Para.param.simple_CI_mode,new_Para.param.NI_OL_mode,
                                                                      self.predict_fc_unit_nums, self.lr,new_Para.param.topK).replace(',', '_')
        self.model_dir = os.path.join(CI_model_dir,self.simple_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        """
        self.model_name区分模型框架，返回的name用于记录在evalute中，区分不同的模型，架构
        :return:
        """
        return self.simple_name


    def get_model(self):
        if self.model is None:
            mashup_CI_input = Input(shape=(self.CI_dim,), dtype='float32', name='mashup_CI_input')
            mashup_NI_input = Input(shape=(self.NI_dim,), dtype='float32', name='mashup_NI_input')
            api_CI_input = Input(shape=(self.CI_dim,), dtype='float32', name='api_CI_input')
            api_NI_input = Input(shape=(self.NI_dim,), dtype='float32', name='api_NI_input')

            inputs = [mashup_CI_input, mashup_NI_input, api_CI_input,api_NI_input]
            predict_vector = Concatenate()(inputs)

            for index, unit_num in enumerate(self.predict_fc_unit_nums):
                predict_vector = Dense(unit_num, name='predict_dense_{}'.format(index), activation='relu',kernel_regularizer=l2(new_Para.param.l2_reg))(predict_vector)
            predict_vector = Dropout(0.5)(predict_vector)

            if new_Para.param.final_activation == 'softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(predict_vector)
            elif new_Para.param.final_activation == 'sigmoid':
                predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)
            print('build PNCF model,done!')

            self.model = Model(inputs=inputs,outputs=[predict_result],name='PNCF')
            if not os.path.exists(self.model_name_path):
                with open(self.model_name_path, 'w+') as f:
                    f.write(self.get_name())
        return self.model

    def get_instances(self, mashup_id_list, api_id_list, mashup_slt_apis_list=None,test_phase_flag=True):
        mashup_CI_features = np.array([self.mashup_CI_features[m_id]for m_id in mashup_id_list])
        api_CI_features = np.array([self.api_CI_features[a_id] for a_id in api_id_list])
        mashup_NI_features = np.array([self.NI_recommend_model.mid_sltAids_2NI_feas[(mashup_id_list[i], tuple(mashup_slt_apis_list[i]))] for i in range(len(mashup_id_list))])
        api_NI_features = np.array([self.NI_recommend_model.i_factors_matrix[a_id] for a_id in api_id_list])
        return (mashup_CI_features,mashup_NI_features,api_CI_features,api_NI_features)

    def save_sth(self):
        pass


# Deep Interest Network Based API Recommendation Approach for Mashup CreationDIN
# 就是先把mashup/service的各种特征拼接，然后DIN，并使用MLP整合 #
class DINRec_model(object):
    def __init__(self, CI_recommend_model=None, CI_model=None, CI_model_dir=None, NI_recommend_model=None,
                 NI_model=None, new_old='new',model_name = 'DINRec'):
        if CI_recommend_model is None and CI_model_dir is None:
            raise ValueError('CI_recommend_model and CI_model_dir cannot be None at the same time!')

        # 路径在CI下面，依次是CI/top_MLP
        if CI_model_dir is None and CI_recommend_model is not None:
            CI_model_dir = CI_recommend_model.model_dir

        self.model_name = model_name
        self.new_old = new_old
        self.CI_dim = 50 * 2
        self.NI_dim = 25
        self.full_dim = self.CI_dim + self.NI_dim
        self.predict_fc_unit_nums = new_Para.param.predict_fc_unit_nums

        self.model = None
        self.CI_recommend_model = CI_recommend_model
        self.NI_recommend_model = NI_recommend_model
        self.CI_model = CI_model
        self.NI_model = NI_model

        self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features = \
            self.CI_recommend_model.get_mashup_api_features(self.CI_recommend_model.all_mashup_num,
                                                            self.CI_recommend_model.all_api_num+1) # 最后一个是填充虚拟api的特征

        self.mashup_CI_features = np.hstack((np.array(self.mashup_texts_features), np.array(self.mashup_tag_features)))
        self.api_CI_features = np.hstack((np.array(self.api_texts_features), np.array(self.api_tag_features)))
        del self.CI_recommend_model.model

        self.lr = new_Para.param.CI_learning_rate  # 暂时用CI的
        self.optimizer = Adam(lr=self.lr)

        self.simple_name = '{}_CI_{}_INI_{}_topMLP{}{}top{}_2'.format(self.model_name,new_Para.param.simple_CI_mode,
                                                                        new_Para.param.NI_OL_mode,
                                                                        self.predict_fc_unit_nums, self.lr,
                                                                        new_Para.param.topK).replace(',', '_')
        self.model_dir = os.path.join(CI_model_dir, self.simple_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        """
        self.model_name区分模型框架，返回的name用于记录在evalute中，区分不同的模型，架构
        :return:
        """
        return self.simple_name

    def get_model(self):
        if self.model is None:
            mashup_input = Input(shape=(self.full_dim,), dtype='float32', name='mashup_input')
            api_input = Input(shape=(self.full_dim,), dtype='float32', name='api_input')
            slt_api_num = new_Para.param.slt_item_num
            slt_apis_input = Input(shape=(slt_api_num,self.full_dim,), dtype='float32', name='slt_apis_input') # (3,125)

            inputs = [mashup_input, api_input, slt_apis_input]
            if self.model_name == 'MLP_embedding':
                x = Lambda(lambda x: tf.expand_dims(x, axis=3))(slt_apis_input)  # (?,3,125,1)
                x = AveragePooling2D(pool_size=(new_Para.param.slt_item_num, 1))(x)
                slt_api_implict_embeddings = Reshape((self.full_dim,))(x)

            elif self.model_name == 'DINRec':
                # whole_attention_model = MultiHeadAttention(slt_api_num, self.full_dim,2,name='implict_MH_whole_')

                whole_attention_model = new_attention_3d_block(slt_api_num, self.full_dim, 'whole_')  # 用单头
                slt_api_implict_embeddings = whole_attention_model([api_input, slt_apis_input, slt_apis_input])

            predict_vector = Concatenate()([mashup_input,api_input,slt_api_implict_embeddings])

            for index, unit_num in enumerate(self.predict_fc_unit_nums):
                predict_vector = Dense(unit_num, name='predict_dense_{}'.format(index), activation='relu',
                                       kernel_regularizer=l2(new_Para.param.l2_reg))(predict_vector)
            predict_vector = Dropout(0.5)(predict_vector)

            if new_Para.param.final_activation == 'softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(predict_vector)
            elif new_Para.param.final_activation == 'sigmoid':
                predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform',
                                       name="prediction")(predict_vector)
            print('build DIN_rec model,done!')

            self.model = Model(inputs=inputs, outputs=[predict_result], name='PNCF')
            if not os.path.exists(self.model_name_path):
                with open(self.model_name_path, 'w+') as f:
                    f.write(self.get_name())
        return self.model

    def get_instances(self, mashup_id_list, api_id_list, mashup_slt_apis_list=None, test_phase_flag=True):
        mashup_CI_features = np.array([self.mashup_CI_features[m_id] for m_id in mashup_id_list])
        api_CI_features = np.array([self.api_CI_features[a_id] for a_id in api_id_list])
        mashup_NI_features = np.array([self.NI_recommend_model.mid_sltAids_2NI_feas[(mashup_id_list[i], tuple(mashup_slt_apis_list[i]))] for i in range(len(mashup_id_list))])
        api_NI_features = np.array([self.NI_recommend_model.i_factors_matrix[a_id] for a_id in api_id_list])

        mashup_slt_apis_array = np.ones((len(mashup_id_list), new_Para.param.slt_item_num),dtype='int32') * self.CI_recommend_model.all_api_num  # default setting,api_num
        for i in range(len(mashup_slt_apis_list)):
            for j in range(len(mashup_slt_apis_list[i])):
                mashup_slt_apis_array[i][j] = mashup_slt_apis_list[i][j]

        # 内容+NI
        mashup_features = np.hstack((mashup_CI_features,mashup_NI_features))
        api_features = np.hstack((api_CI_features, api_NI_features))
        slt_apis_features = np.array([[np.hstack((self.api_CI_features[slt_api],self.NI_recommend_model.i_factors_matrix[slt_api])) for slt_api in slt_apis] for slt_apis in mashup_slt_apis_array])
        return (mashup_features, api_features,slt_apis_features)

    def save_sth(self):
        pass
