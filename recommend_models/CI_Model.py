import os
import pickle
import sys
sys.path.append("..")

from recommend_models.recommend_Model import gx_model
from recommend_models.sequence import SequencePoolingLayer, DNN, AttentionSequencePoolingLayer
from recommend_models.utils import concat_func, NoMask

# from run_deepCTR.run_MISR_deepFM import transfer_testData2
from main.new_para_setting import new_Para
from main.dataset import meta_data, dataset
from Helpers.util import cos_sim, save_2D_list

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dropout,Lambda, Reshape
from tensorflow.python.keras.layers import Dense, Input, Concatenate, Embedding, Multiply,AveragePooling1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam


def slice(x, index):  # 三维的切片
    return x[:, index, :]


class CI_Model (gx_model):
    def __init__(self,old_new):
        super (CI_Model, self).__init__ ()
        self.pairwise_model = None
        self.predict_model = None
        self.old_new = old_new # 新模型还是旧模型
        self.CI_handle_slt_apis_mode = new_Para.param.CI_handle_slt_apis_mode
        self.lr = new_Para.param.CI_learning_rate # 内容部分学习率
        self.optimizer = Adam(lr=self.lr)

        self.set_simple_name()
        self.set_paths()

    def set_simple_name(self):
        self.simple_name = 'new_func_' if self.old_new == 'new' else 'old_func_'  # 新情景，只用功能
        self.simple_name += new_Para.param.text_extracter_mode
        if new_Para.param.text_extracter_mode =='inception':
            self.simple_name += '_'
            self.simple_name += new_Para.param.inception_pooling

        if not new_Para.param.if_inception_MLP:
            self.simple_name += '_NO_extract_MLP'
        if self.old_new == 'new':
            if not self.CI_handle_slt_apis_mode:
                self.simple_name += '_noSlt'
            else:
                self.simple_name += ('_'+self.CI_handle_slt_apis_mode) # 处理方式是全连接还是attention

        self.simple_name += '_{}'.format(self.lr) # 学习率
        if self.old_new == 'new':
            self.simple_name += '_'+ new_Para.param.final_activation # + '_New' # 新模型可选sigmoid和softmax

        # self.simple_name += '_2'
        # self.simple_name += '_newPartTarget' # !!!临时路径！！！效果不好，不用
        # self.simple_name += '_reductData'  # !!!

    def set_paths(self):
        # 路径设置
        self.model_dir = dataset.crt_ds.model_path.format(self.get_simple_name())  # 模型路径
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')
        self.CI_features_path = os.path.join(self.model_dir, 'CI_features.fea')
        self.train_slt_apis_mid_features_path = os.path.join(self.model_dir, 'train_slt_apis_mid_features.csv')
        self.test_slt_apis_mid_features_path = os.path.join(self.model_dir, 'test_slt_apis_mid_features.csv')
        self.ma_text_tag_feas_path = os.path.join(self.model_dir, 'mashup_api_text_tag_feas.dat')  # mashup和api的提取的文本特征

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        """
        self.model_name区分模型框架，返回的name用于记录在evalute中，区分不同的模型，架构
        :return:
        """
        return self.model_name

    def set_attribute(self):
        self.all_mashup_num = meta_data.mashup_num
        self.all_api_num = meta_data.api_num
        self.his_m_ids = dataset.crt_ds.his_mashup_ids

    def set_embedding_matrixs(self):
        # 把每个mashup和api的文本和tag编码，得到四个矩阵，输入id即可映射到这些embedding上,节省get_instaces部分的内存
        if self.encoded_texts is None:
            self.process_texts() # 首先对数据padding，编码处理
        if self.encoded_tags is None:
            self.process_tags()

        # text
        self.mid2text_wordindex = np.array(self.encoded_texts.get_texts_in_index(range(self.all_mashup_num), 'keras_setting', 0))
        self.aid2text_wordindex = np.array(self.encoded_texts.get_texts_in_index(range(self.all_api_num), 'keras_setting', self.all_mashup_num))

        # tag
        self.mid2tag_wordindex = np.array(self.encoded_tags.get_texts_in_index(range(self.all_mashup_num), 'keras_setting', 0))
        self.aid2tag_wordindex = np.array(self.encoded_tags.get_texts_in_index(range(self.all_api_num), 'keras_setting', self.all_mashup_num))
        # instances中slt apis不满足数目时，需要用value(api_num)填充，最终全部映射为0
        self.aid2text_wordindex = np.vstack((self.aid2text_wordindex,np.zeros((1,new_Para.param.MAX_SEQUENCE_LENGTH), dtype=int))).astype(int)
        self.aid2tag_wordindex = np.vstack((self.aid2tag_wordindex, np.zeros((1, new_Para.param.MAX_SEQUENCE_LENGTH), dtype=int))).astype(int)

        # print('mid2text_wordindex[0]:',self.mid2text_wordindex[0])
        # print('mid2tag_wordindex[0:5]:', self.mid2tag_wordindex[0:5])
        # print('aid2text_wordindex[0]:',self.aid2text_wordindex[0])
        # print('aid2tag_wordindex[0:5]:', self.aid2tag_wordindex[0:5])

    def set_embedding_layers(self):
        # embedding
        # using the embedding layer instead of packing in the instance, to save memory
        self.mid2text_embedding_layer = Embedding(self.all_mashup_num, new_Para.param.MAX_SEQUENCE_LENGTH,
                                             embeddings_initializer=Constant(self.mid2text_wordindex),
                                             mask_zero=False, input_length=1,
                                             trainable=False, name='mashup_text_encoding_embedding_layer')

        self.aid2text_embedding_layer = Embedding(self.all_api_num + 1, new_Para.param.MAX_SEQUENCE_LENGTH,
                                             embeddings_initializer=Constant(self.aid2text_wordindex),
                                             mask_zero=False,
                                             trainable=False, name='api_text_encoding_embedding_layer')

        self.mid2tag_embedding_layer = Embedding(self.all_mashup_num, new_Para.param.MAX_SEQUENCE_LENGTH,
                                            embeddings_initializer=Constant(self.mid2tag_wordindex),
                                            mask_zero=False, input_length=1,
                                            trainable=False, name='mashup_tag_encoding_embedding_layer')

        self.aid2tag_embedding_layer = Embedding(self.all_api_num + 1, new_Para.param.MAX_SEQUENCE_LENGTH,
                                            embeddings_initializer=Constant(self.aid2tag_wordindex),
                                            mask_zero=False,
                                            trainable=False, name='api_tag_encoding_embedding_layer')

        self.user_text_feature_extracter = None
        self.item_text_feature_extracter = None
        self.user_tag_feature_extracter = None
        self.item_tag_feature_extracter = None

    def prepare(self):
        self.set_attribute()
        self.set_embedding_matrixs()
        self.set_embedding_layers()

    def user_text_feature_extractor(self):
        if not self.user_text_feature_extracter:
            user_id_input = Input(shape=(1,), dtype='int32') # , name='mashup_id_input'
            user_text_input = Lambda(lambda x: tf.cast(tf.squeeze(x, axis=1), 'int32'))(self.mid2text_embedding_layer(user_id_input))
            user_text_vec = self.feature_extracter_from_texts()(user_text_input) # (?,50)
            self.user_text_feature_extracter = Model(user_id_input, user_text_vec, name='user_text_feature_extracter')
        return self.user_text_feature_extracter

    # mashup和api的文本特征提取器，区别在于ID到文本编码的embedding矩阵不同，但又要公用相同的word_embedding层和inception特征提取器
    def item_text_feature_extractor(self):
        if not self.item_text_feature_extracter:
            item_id_input = Input(shape=(1,), dtype='int32') # , name='api_id_input'
            item_text_input = Lambda(lambda x: tf.cast(tf.squeeze(x, axis=1), 'int32'))(self.aid2text_embedding_layer(item_id_input))
            item_text_vec = self.feature_extracter_from_texts()(item_text_input) # (?,50)
            self.item_text_feature_extracter = Model(item_id_input, item_text_vec, name='item_text_feature_extracter')
        return self.item_text_feature_extracter

    def user_tag_feature_extractor(self):
        if not self.user_tag_feature_extracter:
            user_id_input = Input(shape=(1,), dtype='int32') # , name='user_id_input'
            user_tag_input = Lambda(lambda x: tf.cast(tf.squeeze(x, axis=1), 'int32'))(self.mid2tag_embedding_layer(user_id_input))
            user_tag_embedding = self.get_text_embedding_layer()(user_tag_input)
            user_tag_vec = Lambda(lambda x: tf.squeeze(x, axis=1))(SequencePoolingLayer('mean', supports_masking=True)(user_tag_embedding))
            self.user_tag_feature_extracter = Model(user_id_input, user_tag_vec, name='user_tag_feature_extracter')
        return self.user_tag_feature_extracter

    def item_tag_feature_extractor(self):
        if not self.item_tag_feature_extracter:
            item_id_input = Input(shape=(1,), dtype='int32') # , name='api_id_input'
            item_tag_input = Lambda(lambda x: tf.cast(tf.squeeze(x, axis=1), 'int32'))(self.aid2tag_embedding_layer(item_id_input))
            item_tag_embedding = self.get_tag_embedding_layer()(item_tag_input)
            item_tag_vec = Lambda(lambda x: tf.squeeze(x, axis=1))(SequencePoolingLayer('mean', supports_masking=True)(item_tag_embedding))
            self.item_tag_feature_extracter = Model(item_id_input, item_tag_vec, name='item_tag_feature_extracter')
        return self.item_tag_feature_extracter

    def get_model(self):
        if not self.model:
            mashup_id_input = Input(shape=(1,), dtype='int32', name='mashup_id_input')
            api_id_input = Input(shape=(1,), dtype='int32', name='api_id_input')
            inputs = [mashup_id_input, api_id_input]

            user_text_vec = self.user_text_feature_extractor()(mashup_id_input)
            item_text_vec = self.item_text_feature_extractor()(api_id_input)
            user_tag_vec = self.user_tag_feature_extractor()(mashup_id_input)
            item_tag_vec = self.item_tag_feature_extractor()(api_id_input)
            feature_list = [user_text_vec, item_text_vec, user_tag_vec, item_tag_vec]

            if self.old_new == 'LR_PNCF':  # 旧场景，使用GMF形式的双塔模型
                x = Concatenate(name='user_concatenate')([user_text_vec, user_tag_vec])
                y = Concatenate(name='item_concatenate')([item_text_vec, item_tag_vec])
                output = Multiply()([x, y])
                predict_result = Dense(1, activation='sigmoid', use_bias=False, kernel_initializer='lecun_uniform',name="prediction")(output)  # 参数学习权重，非线性
                self.model = Model(inputs=inputs, outputs=[predict_result], name='predict_model')
                return self.model

            elif self.old_new =='new' and self.CI_handle_slt_apis_mode:
                # 已选择的服务
                mashup_slt_apis_input = Input(shape=(new_Para.param.slt_item_num,), dtype='int32', name='slt_api_ids_input')
                mashup_slt_apis_input_3D = Reshape((new_Para.param.slt_item_num,1))(mashup_slt_apis_input)
                # mashup_slt_apis_num_input = Input(shape=(1,), dtype='int32', name='mashup_slt_apis_num_input')
                inputs.append(mashup_slt_apis_input)
                mask = Lambda(lambda x: K.not_equal(x, self.all_api_num))(mashup_slt_apis_input)  # (?, 3) !!!

                # 已选择的服务直接复用item_feature_extractor
                slt_text_vec_list,slt_tag_vec_list=[],[]
                for i in range(new_Para.param.slt_item_num):
                    x = Lambda (slice, arguments={'index': i}) (mashup_slt_apis_input_3D) # (?,1,1)
                    x = Reshape((1,))(x)
                    temp_item_text_vec = self.item_text_feature_extractor()(x)
                    temp_item_tag_vec = self.item_tag_feature_extractor()(x)
                    slt_text_vec_list.append(temp_item_text_vec)
                    slt_tag_vec_list.append(temp_item_tag_vec)

                if self.CI_handle_slt_apis_mode in ('attention','average') :
                    # text和tag使用各自的attention block
                    slt_text_vec_list = [Reshape((1, new_Para.param.embedding_dim))(key_2D) for key_2D in slt_text_vec_list]
                    slt_tag_vec_list = [Reshape((1, new_Para.param.embedding_dim))(key_2D) for key_2D in slt_tag_vec_list]  # 增加了一维  eg:[None,50]->[None,1,50]
                    text_keys_embs = Concatenate(axis=1)(slt_text_vec_list)  # [?,3,50]
                    tag_keys_embs = Concatenate(axis=1)(slt_tag_vec_list)  # [?,3,50]

                    if self.CI_handle_slt_apis_mode == 'attention':
                        query_item_text_vec = Lambda(lambda x: tf.expand_dims(x, axis=1))(item_text_vec) # (?, 50)->(?, 1, 50)
                        query_item_tag_vec = Lambda(lambda x: tf.expand_dims(x, axis=1))(item_tag_vec)
                        # 压缩历史，得到向量
                        text_hist = AttentionSequencePoolingLayer(supports_masking=True)([query_item_text_vec, text_keys_embs],mask = mask)
                        tag_hist = AttentionSequencePoolingLayer(supports_masking=True)([query_item_tag_vec, tag_keys_embs],mask = mask)

                    else: # 'average'
                        text_hist = SequencePoolingLayer('mean', supports_masking=True)(text_keys_embs,mask = mask)
                        tag_hist = SequencePoolingLayer('mean', supports_masking=True)(tag_keys_embs,mask = mask)

                    text_hist = Lambda(lambda x: tf.squeeze(x, axis=1))(text_hist) # (?, 1, 50)->(?, 50)
                    tag_hist = Lambda(lambda x: tf.squeeze(x, axis=1))(tag_hist)

                elif self.CI_handle_slt_apis_mode == 'full_concate':
                    text_hist = Concatenate(axis=1)(slt_text_vec_list)  # [?,150]
                    tag_hist = Concatenate(axis=1)(slt_tag_vec_list)  # [?,150]
                else:
                    raise ValueError('wrong CI_handle_slt_apis_mode!')

                feature_list.extend([text_hist,tag_hist])

            else: # 包括新模型不处理已选择服务和旧模型
                pass
            feature_list = list(map(NoMask(), feature_list)) # DNN不支持mak，所以不能再传递mask
            all_features = Concatenate(name = 'all_content_concatenate')(feature_list)

            output = DNN(self.content_fc_unit_nums[:-1])(all_features)
            output = Dense (self.content_fc_unit_nums[-1], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg), name='text_tag_feature_extracter') (output)

            # 输出层
            if new_Para.param.final_activation=='softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(output)
            elif new_Para.param.final_activation=='sigmoid':
                predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (output)

            # Model
            # if self.IfUniteNI:
            #     inputs.append(user_NI_input)
            self.model = Model (inputs=inputs,outputs=[predict_result],name='predict_model')

            for layer in self.model.layers:
                print(layer.name)
            print('built CI model, done!')
        return self.model

    def show_text_tag_features(self, train_data, show_num=10):
        """
        检查生成的mashup和api的text和tag的特征是否正常
        """
        if self.old_new == 'old':
            m_ids,a_ids = train_data[:-1]
            instances_tuple = self.get_instances(m_ids[:show_num],a_ids[:show_num])
        elif self.old_new == 'new':
            m_ids,a_ids,slt_a_ids = train_data[:-1]
            instances_tuple = self.get_instances(m_ids[:show_num],a_ids[:show_num],slt_a_ids[:show_num])

        text_tag_middle_model = Model(inputs=[*self.model.inputs],
                                      outputs=[*self.model.get_layer('all_content_concatenate').input[:4]])
        mashup_text_features,apis_text_features, mashup_tag_features,apis_tag_features = text_tag_middle_model.predict([*instances_tuple], verbose=0)

        mashup_text_features_path = os.path.join(self.model_dir,'mashup_text_features.dat')
        apis_text_features_path = os.path.join(self.model_dir,'apis_text_features.dat')
        mashup_tag_features_path = os.path.join(self.model_dir,'mashup_tag_features.dat')
        apis_tag_features_path = os.path.join(self.model_dir,'apis_tag_features.dat')

        save_2D_list(mashup_text_features_path, mashup_text_features, 'a+')
        save_2D_list(apis_text_features_path, apis_text_features, 'a+')
        save_2D_list(mashup_tag_features_path, mashup_tag_features, 'a+')
        save_2D_list(apis_tag_features_path, apis_tag_features, 'a+')

    def get_pairwise_model(self):
        if self.pairwise_model is None:
            if new_Para.param.pairwise and self.model is not None:  # 如果使用pairwise型的目标函数
                mashup_id_input = Input (shape=(1,), dtype='int32', name='mashup_id_input')
                api_id_input = Input (shape=(1,), dtype='int32', name='api_id_input')
                neg_api_id_input = Input(shape=(1,), dtype='int32', name='neg_api_id_input')
                mashup_slt_apis_input = Input(shape=(new_Para.param.slt_item_num,), dtype='int32', name='slt_api_ids_input')
                if self.old_new == 'new':
                    pos_ratings = self.model([mashup_id_input, api_id_input, mashup_slt_apis_input])
                    neg_ratings = self.model([mashup_id_input, neg_api_id_input, mashup_slt_apis_input])  # 再加一个负例api id
                elif self.old_new == 'old':
                    pos_ratings = self.model([mashup_id_input, api_id_input])
                    neg_ratings = self.model([mashup_id_input, neg_api_id_input])

                loss = Lambda(lambda x: K.relu(new_Para.param.margin + x[0] - x[1]),name='sub_result')([neg_ratings, pos_ratings])

                # 注意输入格式！
                if self.old_new == 'new':
                    self.pairwise_model = Model(inputs=[mashup_id_input, api_id_input, mashup_slt_apis_input, neg_api_id_input],outputs=loss)
                elif self.old_new == 'old':
                    self.pairwise_model = Model(inputs=[mashup_id_input, api_id_input, neg_api_id_input], outputs=loss)

                for layer in self.pairwise_model.layers:
                    print(layer.name)
                # # 复用的是同一个对象！
                # print(self.pairwise_model.get_layer('predict_model'),id(self.pairwise_model.get_layer('predict_model')))
                # print(self.model,id(self.model))
        return self.pairwise_model

    def get_instances(self, mashup_id_instances, api_id_instances, slt_api_ids_instances=None, neg_api_id_instances=None,if_Train=False,test_phase_flag=True):
        """
        生成该模型需要的样本
        slt_api_ids_instances是每个样本中，已经选择的api的id序列  变长二维序列
        train和test样例都可用  但是针对一维列表形式，所以测试时先需拆分（text的数据是二维列表）！！！
        :param args:
        :return:
        """
        examples = [np.array(mashup_id_instances), np.array(api_id_instances)]
        # if new_Para.param.need_slt_apis and slt_api_ids_instances: # 是否加入slt_api_ids_instances
        if self.old_new == 'new' and self.CI_handle_slt_apis_mode: # 根据模型变化调整决定，同时输入的数据本身也是对应的
            # 节省内存版, 不够slt_item_num的要padding
            instance_num = len(slt_api_ids_instances)
            padded_slt_api_instances = np.ones((instance_num, new_Para.param.slt_item_num)) * self.all_api_num
            for index1 in range(instance_num):
                a_slt_api_ids = slt_api_ids_instances[index1]
                for index2 in range(len(a_slt_api_ids)):
                    padded_slt_api_instances[index1][index2] = a_slt_api_ids[index2]
            examples.append(padded_slt_api_instances)

        if new_Para.param.pairwise and if_Train: # pair test时，不需要neg_api_id_instances
            examples.append(np.array(neg_api_id_instances))

        examples_tuples = tuple(examples)
        return examples_tuples

    def get_mashup_api_features(self,mashup_num,api_num):
        """
        得到每个mashup和api经过特征提取器或者平均池化得到的特征，可以直接用id索引，供构造instance的文本部分使用
        :param text_tag_recommend_model:
        :param mashup_num:
        :param api_num:
        :return:
        """
        if os.path.exists(self.ma_text_tag_feas_path):
            with open(self.ma_text_tag_feas_path,'rb') as f1:
                mashup_texts_features, mashup_tag_features, api_texts_features, api_tag_features = pickle.load(f1)
        else:
            # 前四个分别是 user_text_vec, item_text_vec, user_tag_vec, item_tag_vec
            text_tag_middle_model = Model(inputs=[*self.model.inputs[:2]],
                                          outputs=[self.model.get_layer('all_content_concatenate').input[0],
                                                   self.model.get_layer('all_content_concatenate').input[1],
                                                   self.model.get_layer('all_content_concatenate').input[2],
                                                   self.model.get_layer('all_content_concatenate').input[3]])

            feature_mashup_ids = list(np.unique([m_id for m_id in range (mashup_num)]))
            feature_instances_tuple = self.get_instances(feature_mashup_ids, [0] * len(feature_mashup_ids))
            mashup_texts_features,_1, mashup_tag_features,_2 = text_tag_middle_model.predict ([*feature_instances_tuple], verbose=0)

            feature_api_ids = list(np.unique([a_id for a_id in range (api_num)]))
            feature_instances_tuple = self.get_instances([0] * len(feature_api_ids),feature_api_ids)
            _1,api_texts_features, _2,api_tag_features = text_tag_middle_model.predict ([*feature_instances_tuple], verbose=0)

            with open(self.ma_text_tag_feas_path, 'wb') as f2:
                pickle.dump((mashup_texts_features, mashup_tag_features, api_texts_features, api_tag_features),f2)
        return mashup_texts_features,mashup_tag_features,api_texts_features,api_tag_features


