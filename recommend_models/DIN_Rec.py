import os

from main.dataset import meta_data, dataset
from main.new_para_setting import new_Para
from recommend_models.CI_Model import CI_Model
from recommend_models.sequence import DNN, AttentionSequencePoolingLayer
from recommend_models.utils import concat_func, NoMask

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Dense, Input, Concatenate, Embedding
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam

class DIN_Rec (object):
    def __init__(self,CI_recommend_model,predict_fc_unit_nums,new_old = 'new'):
        super(DIN_Rec, self).__init__()
        self.new_old = new_old
        self.CI_recommend_model = CI_recommend_model # 基于哪个CI模型得到的text/tag特征
        self.predict_fc_unit_nums = predict_fc_unit_nums
        self.model = None
        self.lr = new_Para.param.CI_learning_rate # 内容部分学习率
        self.optimizer = Adam(lr=self.lr)
        self.set_simple_name()
        self.set_model_name()
        self.set_paths()

    def set_simple_name(self):
        self.simple_name = 'DIN_Rec' + '_DNN{}'.format(self.predict_fc_unit_nums).replace(',', '_')

    def set_model_name(self):
        self.model_name = self.simple_name

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        return self.model_name

    def set_paths(self):
        # 路径设置
        self.model_dir = os.path.join(self.CI_recommend_model.model_dir,self.get_simple_name())  # 模型路径
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')

    def set_attribute(self):
        self.all_mashup_num = meta_data.mashup_num
        self.all_api_num = meta_data.api_num
        self.his_m_ids = dataset.crt_ds.his_mashup_ids

        self.text_fea_dim = new_Para.param.inception_fc_unit_nums[-1]
        self.tag_fea_dim = new_Para.param.embedding_dim
        self.NI_emb_dim = new_Para.param.num_feat

    def set_embedding_matrixs(self):
        # text,tag
        self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features = \
            self.CI_recommend_model.get_mashup_api_features(self.all_mashup_num, self.all_api_num)
        # api 需要增加一个全为0的，放在最后，id为api_num，用来对slt_apis填充
        self.api_tag_features = np.vstack((self.api_tag_features, np.zeros((1, new_Para.param.embedding_dim))))
        self.api_texts_features = np.vstack((self.api_texts_features, np.zeros((1, new_Para.param.inception_fc_unit_nums[-1]))))

        # API embedding
        self.i_factors_matrix = np.zeros((meta_data.api_num + 1, new_Para.param.num_feat))
        a_embeddings_array = np.array(dataset.UV_obj.a_embeddings)
        for id, index in dataset.UV_obj.a_id2index.items():
            self.i_factors_matrix[id] = a_embeddings_array[index]

    def set_embedding_layers(self):
        self.mid2text_fea_layer = Embedding(self.all_mashup_num, self.text_fea_dim,
                                             embeddings_initializer=Constant(self.mashup_texts_features),
                                             mask_zero=False, input_length=1,
                                             trainable=False, name='mashup_text_fea_layer')

        self.aid2text_fea_layer = Embedding(self.all_api_num + 1, self.tag_fea_dim,
                                             embeddings_initializer=Constant(self.api_texts_features),
                                             mask_zero=False,
                                             trainable=False, name='api_text_fea_layer')

        self.mid2tag_fea_layer = Embedding(self.all_mashup_num, self.text_fea_dim,
                                            embeddings_initializer=Constant(self.mashup_tag_features),
                                            mask_zero=False, input_length=1,
                                            trainable=False, name='mashup_tag_fea_layer')

        self.aid2tag_fea_layer = Embedding(self.all_api_num + 1, self.tag_fea_dim,
                                            embeddings_initializer=Constant(self.api_tag_features),
                                            mask_zero=False,
                                            trainable=False, name='api_tag_fea_layer')

        self.api_implict_emb_layer = Embedding(self.all_api_num + 1,
                                                new_Para.param.num_feat,
                                                embeddings_initializer=Constant(self.i_factors_matrix),
                                                mask_zero=False,
                                                trainable=False,
                                                name='api_implict_emb_layer')

    def prepare(self):
        self.set_attribute()
        self.set_embedding_matrixs()
        self.set_embedding_layers()

    def get_model(self):
        if not self.model:
            mashup_id_input = Input(shape=(1,), dtype='int32', name='mashup_id_input')
            api_id_input = Input(shape=(1,), dtype='int32', name='api_id_input')
            inputs = [mashup_id_input, api_id_input]

            mashup_text_fea = self.mid2text_fea_layer(mashup_id_input)  # (None,1,25)
            api_text_fea = self.aid2text_fea_layer(api_id_input)  # (None,1,25)

            mashup_tag_fea = self.mid2tag_fea_layer(mashup_id_input)  # (None,1,25)
            api_tag_fea = self.aid2tag_fea_layer(api_id_input)  # (None,1,25)

            api_implict_emb = self.api_implict_emb_layer(api_id_input)  # (None,1,25)

            feature_list = [mashup_text_fea, api_text_fea, mashup_tag_fea, api_tag_fea, api_implict_emb]

            if self.new_old == 'new' and new_Para.param.need_slt_apis:
                mashup_slt_apis_input = Input(shape=(new_Para.param.slt_item_num,), dtype='int32', name='slt_api_ids_input')
                inputs.append(mashup_slt_apis_input)

                keys_slt_api_text_feas = self.aid2text_fea_layer(mashup_slt_apis_input)  # (None,3,25)
                keys_slt_api_tag_feas = self.aid2tag_fea_layer(mashup_slt_apis_input)  # (None,3,25)
                keys_slt_api_implict_embs = self.api_implict_emb_layer(mashup_slt_apis_input)  # (None,3,25)

                mask = Lambda(lambda x: K.not_equal(x, self.all_api_num))(mashup_slt_apis_input)  # (?, 3) !!!

                # query_api_text_vec = Lambda(lambda x: tf.expand_dims(x, axis=1))(api_text_fea)  # (?, 50)->(?, 1, 50)
                # query_api_tag_vec = Lambda(lambda x: tf.expand_dims(x, axis=1))(api_tag_fea)
                # query_api_implict_emb = Lambda(lambda x: tf.expand_dims(x, axis=1))(api_implict_emb)

                # 压缩历史，得到向量  ->(?, 1, 50)
                text_hist = AttentionSequencePoolingLayer(supports_masking=True)([api_text_fea, keys_slt_api_text_feas],mask=mask)
                tag_hist = AttentionSequencePoolingLayer(supports_masking=True)([api_tag_fea, keys_slt_api_tag_feas],mask=mask)
                implict_emb_hist = AttentionSequencePoolingLayer(supports_masking=True)([api_implict_emb, keys_slt_api_implict_embs],mask=mask)

                feature_list = [mashup_text_fea,api_text_fea,text_hist,mashup_tag_fea,api_tag_fea,tag_hist,api_implict_emb,implict_emb_hist]
                feature_list = list(map(NoMask(), feature_list)) # DNN不支持mak，所以不能再传递mask

            all_features = Concatenate(name = 'all_content_concatenate')(feature_list)
            all_features = Lambda(lambda x: tf.squeeze(x, axis=1)) (all_features)

            output = DNN(self.predict_fc_unit_nums[:-1])(all_features)
            output = Dense (self.predict_fc_unit_nums[-1], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (output)

            # 输出层
            if new_Para.param.final_activation=='softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(output)
            elif new_Para.param.final_activation=='sigmoid':
                predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (output)

            self.model = Model (inputs=inputs,outputs=[predict_result],name='predict_model')
        return self.model

    # get_instance()直接复用CI的
    def get_instances(self, mashup_id_instances, api_id_instances, slt_api_ids_instances=None,test_phase_flag=True):
        examples = [np.array(mashup_id_instances), np.array(api_id_instances)]
        if new_Para.param.need_slt_apis and slt_api_ids_instances: # 是否加入slt_api_ids_instances
            # 节省内存版, 不够slt_item_num的要padding
            instance_num = len(slt_api_ids_instances)
            padded_slt_api_instances = np.ones((instance_num, new_Para.param.slt_item_num)) * self.all_api_num
            for index1 in range(instance_num):
                a_slt_api_ids = slt_api_ids_instances[index1]
                for index2 in range(len(a_slt_api_ids)):
                    padded_slt_api_instances[index1][index2] = a_slt_api_ids[index2]
            examples.append(padded_slt_api_instances)

        examples_tuples = tuple(examples)
        return examples_tuples
