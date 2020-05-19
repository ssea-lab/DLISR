import os
import pickle
import sys
import time

from keras.optimizers import Adam

from recommend_models.attention_block import attention, attention_3d_block

sys.path.append("..")
from main.new_para_setting import new_Para
from main.dataset import meta_data, dataset
from keras.regularizers import l2
from recommend_models.HIN_sim import cpt_p13_sim, word_sim, cpt_content_sim, cpt_p46_sim, mashup_HIN_sims

from recommend_models.text_tag_model import gx_text_tag_model
from Helpers.util import cos_sim, save_2D_list
import numpy as np
import tensorflow as tf
from keras.layers.core import Dropout, Lambda, Reshape
from keras.layers import Dense, Input, concatenate, Concatenate, Embedding, Multiply
from keras.models import Model
from main.processing_data import process_data, get_mashup_api_allCategories, get_mashup_api_field

from keras import backend as K
from keras.initializers import Constant


def slice(x, index):  # 三维的切片
    return x[:, index, :]

class gx_text_tag_continue_model (gx_text_tag_model):
    def __init__(self,old_new):
        super (gx_text_tag_continue_model, self).__init__ ()
        self.old_new = old_new # 新模型还是旧模型
        self.CI_handle_slt_apis_mode = new_Para.param.CI_handle_slt_apis_mode
        self.simple_name = 'new_func' if self.old_new == 'new' else 'old_func'  # 新情景，只用功能
        self.simple_name += '_'
        self.simple_name += new_Para.param.text_extracter_mode
        if new_Para.param.text_extracter_mode =='inception':
            self.simple_name += '_'
            self.simple_name += new_Para.param.inception_pooling

        if not new_Para.param.if_inception_MLP:
            self.simple_name += '_NO_extract_MLP' # 特征提取器中不适用MLP
        if self.old_new == 'new':
            self.simple_name += ('_'+self.CI_handle_slt_apis_mode) # 处理方式是全连接还是attention

        self.lr = new_Para.param.CI_learning_rate # 内容部分学习率
        self.optimizer = Adam(lr=self.lr)
        self.simple_name += '_CIlr_{}'.format(self.lr) # 跟学习率！！！
        self.model_dir = dataset.crt_ds.model_path.format(self.get_simple_name()) # 模型路径

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        # self.model_name 不变
        name = self.simple_name+ self.model_name
        return name

    def get_text_tag_part(self, user_text_input, item_text_input,user_categories_input, item_categories_input, slt_items_texts_input=None,slt_items_categories_input=None,
                          user_tag_nums_input=None,item_tag_nums_input=None,slt_item_tag_nums_input=None):
        """
        同时处理text和tag;新增平均tag的处理
        :param user_text_input:
        :param item_text_input:
        :return:
        """
        # text
        user_text_feature = self.feature_extracter_from_texts () (user_text_input)  # (None,embedding)
        print(user_text_feature)
        item_text_feature = self.feature_extracter_from_texts () (item_text_input)

        # tags
        if new_Para.param.tag_manner == 'new_average':
            print('***in: user_categories_input',user_categories_input)
            print('***in: user_categories_input', user_tag_nums_input)
            user_categories_feature = self.get_categories_feature_extracter () ([user_categories_input,user_tag_nums_input])
            item_categories_feature = self.get_categories_feature_extracter () ([item_categories_input,item_tag_nums_input])
        else:
            user_categories_feature = self.get_categories_feature_extracter () (user_categories_input)
            item_categories_feature = self.get_categories_feature_extracter () (item_categories_input)
        print('user_categories_feature:',user_categories_feature)

        if self.old_new =='new':
            slt_text_feature_list=[]
            for i in range(new_Para.param.slt_item_num):
                x = Lambda (slice, output_shape=(1, 150), arguments={'index': i}) (slt_items_texts_input) # (None,150)
                print(x) # (None,150)

                a_feature= self.feature_extracter_from_texts () (x) # (None,50)
                print(a_feature) # shape=(?, 50)
                slt_text_feature_list.append(a_feature)

            slt_tag_feature_list=[]
            for i in range(new_Para.param.slt_item_num):
                x = Lambda (slice, output_shape=(1, 150), arguments={'index': i}) (slt_items_categories_input) # (None,150)
                print('x',x) # shape=(?, 150), dtype=int32
                if new_Para.param.tag_manner == 'new_average':
                    slt_item_tag_num = Lambda(slice, output_shape=(1,1), arguments={'index': i})(slt_item_tag_nums_input)
                    print('before divide,slt_item_tag_num:',slt_item_tag_num) # shape=(?, 1), dtype=float32

                    print('***in2: x', x)
                    print('***in2: slt_item_tag_num', slt_item_tag_num)
                    a_feature = self.get_categories_feature_extracter()([x, slt_item_tag_num]) # (None,50)
                    print('after divide,slt_item_tag_num:', a_feature)
                else:
                    a_feature = self.get_categories_feature_extracter()(x)  # (None,50)?
                    print(a_feature)
                slt_tag_feature_list.append(a_feature)

            # slt_api text/tag feature的整合：直接拼接还是attention？
            if self.CI_handle_slt_apis_mode == 'full_concate':
                concated_slt_items_texts_feature= Concatenate(name='slt_texts_concat',axis=1) (slt_text_feature_list) #
                print(concated_slt_items_texts_feature) # (?,150)
                concated_slt_items_categories_feature= Concatenate(name='slt_tags_concat',axis=1) (slt_tag_feature_list)

            if self.CI_handle_slt_apis_mode=='attention':
                # text_feature_size = slt_text_feature_list[0].shape[1].value # int 50  slt数目可能为0？
                slt_text_feature_list = [Reshape((1, new_Para.param.embedding_dim))(key_2D) for key_2D in slt_text_feature_list]  # 增加了一维  eg:[None,50]->[None,1,50]
                key1 = Concatenate(axis=1)(slt_text_feature_list)  # eg:[None,3,50]

                slt_tag_feature_list = [Reshape((1, new_Para.param.embedding_dim))(key_2D) for key_2D in slt_tag_feature_list]  # 增加了一维  eg:[None,50]->[None,1,50]
                key2 = Concatenate(axis=1)(slt_tag_feature_list)  # eg:[None,3,50]

                # # 把文本和tag拼接作为特征无意义
                # key = Concatenate(axis=-1)([key1,key2]) # 内容特征由text和tag拼接 eg:[None,3,100]
                # query = Concatenate(axis=-1)([item_text_feature,item_categories_feature]) #@@@!!!
                # # query = Concatenate(axis=-1)([user_text_feature, user_categories_feature])
                # print('query,', query)
                # print('key,', key)
                # # concated_slt_items_feature = attention(query,key,key)
                # concated_slt_items_feature = attention_3d_block(query, key, key)

                # 先分别attention再拼接
                concated_slt_items_texts_feature = attention_3d_block(item_text_feature, key1, key1,'text_')
                concated_slt_items_categories_feature = attention_3d_block(item_categories_feature, key2, key2,'tag_')
                # concated_slt_items_feature = Concatenate(axis=-1)([text_feature_results,tag_feature_results]) #

        if self.old_new == 'LR_PNCF':  # 使用GMF形式
            x = Concatenate(name='user_concatenate')([user_text_feature,user_categories_feature])
            y= Concatenate(name='item_concatenate') ([item_text_feature, item_categories_feature])
            results = Multiply()([x,y])
            return results # 直接输出最终预测值

        if new_Para.param.merge_manner == 'direct_merge':
            if self.old_new == 'new':
                if self.CI_handle_slt_apis_mode=='full_concate':
                    concated_slt_items_categories_feature = Reshape((new_Para.param.slt_item_num*new_Para.param.embedding_dim,))(concated_slt_items_categories_feature)
                    x = Concatenate (name='all_content_concatenate') ([user_text_feature, item_text_feature, concated_slt_items_texts_feature,
                                                            user_categories_feature,item_categories_feature,concated_slt_items_categories_feature])  # 整合文本和类别特征，尽管层次不太一样
                elif self.CI_handle_slt_apis_mode=='attention':
                    x = Concatenate(name='all_content_concatenate')([user_text_feature, item_text_feature,concated_slt_items_texts_feature,
                         user_categories_feature, item_categories_feature,concated_slt_items_categories_feature])  # 整合文本和类别特征，尽管层次不太一样 concated_slt_items_feature

            elif self.old_new == 'old':
                x = Concatenate(name='all_content_concatenate')(
                    [user_text_feature, item_text_feature,user_categories_feature, item_categories_feature])
        """
        # 暂时跟得到中间层结果的方法有冲突，暂不可用
        elif new_Para.param.merge_manner == 'final_merge':
            if self.old_new == 'new':
                x = concatenate ([user_text_feature, item_text_feature,concated_slt_items_texts_feature])
            elif self.old_new == 'old':
                x = concatenate([user_text_feature, item_text_feature])
            for unit_num in self.text_fc_unit_nums:
                x = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (x)

            if self.old_new == 'new':
                y = concatenate ([user_categories_feature, item_categories_feature,concated_slt_items_categories_feature])
            elif self.old_new == 'old':
                y = concatenate([user_categories_feature, item_categories_feature])
            for unit_num in self.tag_fc_unit_nums:
                y = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (y)

            x = concatenate ([x, y])  # 整合文本和类别特征，尽管层次不太一样
        """

        for index,unit_num in enumerate(self.content_fc_unit_nums[:-1]): # 需要名字！！！
            x = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg),name ='content_dense_{}'.format(index)) (x)

        x = Dense (self.content_fc_unit_nums[-1], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg), name='text_tag_feature_extracter') (x)

        print ('built text and tag layer, done!')
        return x

    def set_attribute(self): # ,all_mashup_num,all_api_num,his_m_ids
        self.all_mashup_num = meta_data.mashup_num
        self.all_api_num = meta_data.api_num
        self.his_m_ids = dataset.crt_ds.his_mashup_ids

    def set_text_encoding_matrixs(self):
        if self.encoded_texts is None:
            self.process_texts() # 首先对数据padding，编码处理

        # 把每个mashup和api的文本和tag编码，得到四个矩阵，输入id即可映射到这些embedding上
        self.mid2text_wordindex = np.array(self.encoded_texts.get_texts_in_index(range(self.all_mashup_num), 'keras_setting', 0))
        self.aid2text_wordindex = np.array(self.encoded_texts.get_texts_in_index(range(self.all_api_num), 'keras_setting', self.num_users))

        # mashup/api的类型信息  二维词汇列表形式
        mashup_id2info = meta_data.pd.get_mashup_api_id2info ('mashup')
        api_id2info = meta_data.pd.get_mashup_api_id2info ('api')
        mashup_categories = [get_mashup_api_allCategories ('mashup', mashup_id2info, mashup_id, new_Para.param.Category_type) for
                             mashup_id in range(self.all_mashup_num)]
        api_categories = [get_mashup_api_allCategories ('api', api_id2info, api_id, new_Para.param.Category_type) for api_id in
                          range(self.all_api_num)]

        self.mid2tag_wordindex = np.array(self.encoded_texts.get_texts_in_index(mashup_categories, 'self_padding'))
        self.aid2tag_wordindex = np.array(self.encoded_texts.get_texts_in_index(api_categories, 'self_padding'))
        # instances中slt apis不满足数目时，需要用api_num值填充，最终全部映射为0
        self.aid2text_wordindex = np.vstack((self.aid2text_wordindex,np.zeros((1,new_Para.param.MAX_SEQUENCE_LENGTH))))
        self.aid2tag_wordindex = np.vstack((self.aid2tag_wordindex, np.zeros((1, new_Para.param.MAX_SEQUENCE_LENGTH))))

        print(' shape of mid2text_wordindex:',self.mid2text_wordindex.shape)
        print(' shape of mid2tag_wordindex:', self.mid2tag_wordindex.shape)
        print(' shape of aid2text_wordindex:',self.aid2text_wordindex.shape)
        print(' shape of aid2tag_wordindex:', self.aid2tag_wordindex.shape)
        print('mid2text_wordindex[0]:',self.mid2text_wordindex[0])
        print('mid2tag_wordindex[0]:', self.mid2tag_wordindex[0])

        if new_Para.param.tag_manner == 'new_average':
            self.mashup_tag_nums = 1/np.array([len(mashup_cates) for mashup_cates in mashup_categories]) #
            api_tag_nums = [len(api_cates) for api_cates in api_categories]
            api_tag_nums.append(1) # 对于padding用的最大id，个数设为1
            self.api_tag_nums = 1/np.array(api_tag_nums) #


    def prepare(self):
        self.set_attribute()
        self.set_text_encoding_matrixs()

    def show_slt_apis_tag_features(self, train_data, show_num=10): # 需要全连接，attention时不适用,待改！！！
        """
        在模型训练之后，观察slt apis的tag特征，150维，填充的全为0，只想关注padding的0的映射情况
        :return:
        """
        if not (new_Para.param.data_mode=='newScene' and new_Para.param.need_slt_apis) :
            m_ids,a_ids = train_data[:-1]
            instances_tuple = self.get_instances(m_ids[:show_num],a_ids[:show_num])
            text_tag_middle_model = Model(inputs=[*self.model.inputs],
                                          outputs=[*self.model.get_layer('all_content_concatenate').input])
            mashup_text_features,apis_text_features, mashup_tag_features,apis_tag_features = text_tag_middle_model.predict([*instances_tuple], verbose=0)

            mashup_text_features_path = os.path.join(self.model_dir,'mashup_text_features.dat')
            apis_text_features_path = os.path.join(self.model_dir,'apis_text_features.dat')
            mashup_tag_features_path = os.path.join(self.model_dir,'mashup_tag_features.dat')
            apis_tag_features_path = os.path.join(self.model_dir,'apis_tag_features.dat')

            save_2D_list(mashup_text_features_path, mashup_text_features, 'a+')
            save_2D_list(apis_text_features_path, apis_text_features, 'a+')
            save_2D_list(mashup_tag_features_path, mashup_tag_features, 'a+')
            save_2D_list(apis_tag_features_path, apis_tag_features, 'a+')

            print('show text and tag_features of mashup, apis when need not slt_apis, done!')
        else: # 只针对full_concate的情景，attention没必要
            if self.CI_handle_slt_apis_mode=='full_concate':
                m_ids,a_ids,slt_a_is = train_data[:-1]
                instances_tuple = self.get_instances(m_ids[:show_num],a_ids[:show_num],slt_a_is[:show_num])
                # instances_tuple = self.get_instances(*train_data[:-1,:show_num]) # 第一维去除tag，第二位选择instances个数  TypeError: list indices must be integers or slices, not tuple
                text_tag_middle_model = Model(inputs=[*self.model.inputs],
                                              outputs=[*self.model.get_layer('all_content_concatenate').input])
                results = text_tag_middle_model.predict([*instances_tuple], verbose=0)
                slt_apis_text_features, slt_apis_tag_features = results[2],results[5]
                slt_text_features_path = os.path.join(self.model_dir, 'slt_apis_text_features.dat')
                slt_tag_features_path = os.path.join(self.model_dir,'slt_apis_tag_features.dat')
                save_2D_list(slt_text_features_path, slt_apis_text_features, 'w+')
                save_2D_list(slt_tag_features_path, slt_apis_tag_features, 'w+')
                print('show text and tag_features of slt_apis when need slt_apis, done!')

    def get_model(self): # 'old' 'new'
        if self.model is None:
            mashup_id_input = Input (shape=(1,), dtype='int32', name='mashup_id_input')
            api_id_input = Input (shape=(1,), dtype='int32', name='api_id_input')

            # get the functional feature input, using the embedding layer instead of packing in the instance, to save memory
            mid2text_embedding_layer = Embedding(self.all_mashup_num,new_Para.param.MAX_SEQUENCE_LENGTH,
                                             embeddings_initializer=Constant(self.mid2text_wordindex),
                                             mask_zero=False, input_length=1,
                                             trainable=False, name = 'mashup_text_encoding_embedding_layer') # Constant()(dtype='int32',shape=(1979, 150)

            aid2text_embedding_layer = Embedding(self.all_api_num+1,new_Para.param.MAX_SEQUENCE_LENGTH,
                                             embeddings_initializer=Constant(self.aid2text_wordindex),
                                             mask_zero=False,
                                             trainable=False, name = 'api_text_encoding_embedding_layer')

            mid2tag_embedding_layer = Embedding(self.all_mashup_num,new_Para.param.MAX_SEQUENCE_LENGTH,
                                             embeddings_initializer=Constant(self.mid2tag_wordindex),
                                             mask_zero=False, input_length=1,
                                             trainable=False, name = 'mashup_tag_encoding_embedding_layer')

            aid2tag_embedding_layer = Embedding(self.all_api_num+1,new_Para.param.MAX_SEQUENCE_LENGTH,
                                             embeddings_initializer=Constant(self.aid2tag_wordindex),
                                             mask_zero=False,
                                             trainable=False, name = 'api_tag_encoding_embedding_layer')

            user_text_input= mid2text_embedding_layer(mashup_id_input)
            item_text_input = aid2text_embedding_layer(api_id_input)
            user_categories_input = mid2tag_embedding_layer(mashup_id_input)
            item_categories_input = aid2tag_embedding_layer(api_id_input)

            user_text_input = Lambda(lambda x: tf.cast(tf.squeeze(x,axis=1),'int32'))(user_text_input)
            item_text_input = Lambda(lambda x: tf.cast(tf.squeeze(x,axis=1), 'int32'))(item_text_input)
            user_categories_input = Lambda(lambda x: tf.cast(tf.squeeze(x,axis=1), 'int32'))(user_categories_input)
            item_categories_input = Lambda(lambda x: tf.cast(tf.squeeze(x,axis=1), 'int32'))(item_categories_input)

            print('user_text_input',user_text_input) # shape=(?, 150), dtype=float32
            print('item_text_input', item_text_input)
            print('user_categories_input', user_categories_input)
            print('item_categories_input', item_categories_input)

            if self.old_new == 'new':
                mashup_slt_apis_input = Input(shape=(new_Para.param.slt_item_num,), dtype='int32', name='slt_api_ids_input')
                slt_items_texts_input = aid2text_embedding_layer(mashup_slt_apis_input)
                slt_items_categories_input = aid2tag_embedding_layer(mashup_slt_apis_input)

                slt_items_texts_input = Lambda(lambda x: tf.cast(x, 'int32'))(slt_items_texts_input)
                slt_items_categories_input = Lambda(lambda x: tf.cast(x, 'int32'))(slt_items_categories_input)

                print('slt_items_texts_input', slt_items_texts_input)  # shape=(?, 3, 150), dtype=float32
                print('slt_items_categories_input', slt_items_categories_input)

            if new_Para.param.tag_manner == 'new_average':
                mid2tag_num_layer = Embedding(self.all_mashup_num, 1,
                                              embeddings_initializer=Constant(self.mashup_tag_nums),
                                              mask_zero=False, input_length=1,
                                              trainable=False, name='mashup_tag_num_layer')
                aid2tag_num_layer = Embedding(self.all_api_num + 1, 1,
                                              embeddings_initializer=Constant(self.api_tag_nums),
                                              mask_zero=False, input_length=1,
                                              trainable=False, name='api_tag_num_layer')

                user_tag_num_input = mid2tag_num_layer(mashup_id_input)  # shape=(?, 1, 1)
                item_tag_num_input = aid2tag_num_layer(api_id_input)

                user_tag_num_input = Lambda(lambda x: tf.squeeze(x, axis=2))(user_tag_num_input)
                item_tag_num_input = Lambda(lambda x: tf.squeeze(x, axis=2))(item_tag_num_input)
                print('item_tag_num_input', item_tag_num_input) # shape=(?, 1), dtype=float32

                if self.old_new == 'new': # 新场景且使用new_average
                    slt_item_tag_num_input = aid2tag_num_layer(mashup_slt_apis_input)  # shape=(?, 3, 1)
                    # slt_item_tag_num_input = Lambda(lambda x: tf.squeeze(x, axis=2))(slt_item_tag_num_input)
                    print('slt_item_tag_num_input', slt_item_tag_num_input)

                    x = self.get_text_tag_part(user_text_input, item_text_input,user_categories_input, item_categories_input,
                                               slt_items_texts_input,slt_items_categories_input,
                                               user_tag_num_input,item_tag_num_input,slt_item_tag_num_input
                                               )
                elif self.old_new == 'old' or self.old_new == 'LR_PNCF':
                    x = self.get_text_tag_part(user_text_input, item_text_input,user_categories_input, item_categories_input,
                                               user_tag_nums_input=user_tag_num_input,item_tag_nums_input=item_tag_num_input
                                               )

            else:
                if self.old_new == 'new':
                    x = self.get_text_tag_part (user_text_input, item_text_input,user_categories_input, item_categories_input,
                                                slt_items_texts_input=slt_items_texts_input,slt_items_categories_input=slt_items_categories_input)
                elif self.old_new == 'old':
                    x = self.get_text_tag_part(user_text_input, item_text_input,user_categories_input, item_categories_input)

            print('x:', x)
            x = Dropout (0.5) (x)

            if self.old_new == 'LR_PNCF':
                predict_result = Dense(1, activation='sigmoid', use_bias=False, kernel_initializer='lecun_uniform', name="prediction")(x)  # 参数学习权重，非线性
            else:
                predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (x)

            if self.old_new == 'new':
                self.model = Model (inputs=[mashup_id_input,api_id_input,mashup_slt_apis_input],
                                    outputs=[predict_result])
            elif self.old_new == 'old' or self.old_new == 'LR_PNCF':
                self.model = Model (inputs=[mashup_id_input,api_id_input],outputs=[predict_result])

            for layer in self.model.layers:
                print(layer.name)
            print ('built whole model, done!')

        print('word embedding layer', np.array(self.text_embedding_layer.get_weights ()).shape)
        print('some embedding parameters:')
        print (self.text_embedding_layer.get_weights ()[0][:2])
        print (self.text_embedding_layer.get_weights ()[0][-1])

        # plot_model (self.model, to_file='text_tag_continue.png', show_shapes=True)
        return self.model

    def get_instances(self, mashup_id_instances, api_id_instances, slt_api_ids_instances=None, mashup_only=False):
        """
        根据get_model_instances得到的mashup_id_instances, api_id_instances生成该模型需要的样本
        slt_api_ids_instances是每个样本中，已经选择的api的id序列  变长二维序列
        train和test样例都可用  但是针对一维列表形式，所以test先需拆分！！！
        :param args:
        :return:
        """
        if mashup_only or not new_Para.param.need_slt_apis: #新场景下的数据用旧模型训练时，不需要slt_apis
            examples = (
                np.array (mashup_id_instances),
                np.array (api_id_instances),
            )
        else:
            # 节省内存版, 不够slt_item_num的要padding
            instance_num=len(slt_api_ids_instances)
            padded_slt_api_instances=np.ones((instance_num,new_Para.param.slt_item_num))*self.all_api_num
            for index1 in range(instance_num):
                a_slt_api_ids= slt_api_ids_instances[index1]
                for index2 in range(len(a_slt_api_ids)):
                    padded_slt_api_instances[index1][index2]= a_slt_api_ids[index2]
            examples = (np.array (mashup_id_instances), np.array (api_id_instances),padded_slt_api_instances)
        return examples

    def get_mashup_text_tag_features(self,mashup_ids):
        """
        传入待测mashup的id列表，返回特征提取器提取的mashup的text和tag的feature
        :param mashup_ids: 可以分别传入train和test的mashup
        :return:
        """
        if self.old_new == 'new':
            index1,index2 = 0,3
        else:
            index1,index2 = 0,2
        text_tag_middle_model = Model (inputs=[self.model.inputs[0], self.model.inputs[1]],
                                       outputs=[self.model.get_layer ('all_content_concatenate').input[index1],
                                                self.model.get_layer ('all_content_concatenate').input[index2]])

        feature_mashup_ids=list(np.unique(mashup_ids))
        feature_instances_tuple = self.get_instances(feature_mashup_ids,[0]*len(feature_mashup_ids), mashup_only=True)
        text_features,tag_features=text_tag_middle_model.predict ([*feature_instances_tuple], verbose=0)
        return text_features,tag_features

    def get_api_text_tag_features(self, api_ids):
        """
        传入待测api_ids的id列表，返回特征提取器提取的api_ids的text和tag的feature
        :param api_ids: 一般直接将所有的api id传入即可
        :return:
        """
        if self.old_new == 'new':
            index1,index2 = 1,4
        else:
            index1,index2 = 1,3
        text_tag_middle_model = Model (inputs=[self.model.inputs[0], self.model.inputs[1]],
                                       outputs=[self.model.get_layer ('all_content_concatenate').input[index1],
                                                self.model.get_layer ('all_content_concatenate').input[index2]])

        feature_api_ids = list (np.unique (api_ids))
        feature_instances_tuple = self.get_instances ([0] * len (feature_api_ids), feature_api_ids,mashup_only=True)
        text_features, tag_features = text_tag_middle_model.predict ([*feature_instances_tuple], verbose=0)
        return text_features, tag_features

    def get_mashup_api_features(self,mashup_num,api_num):
        """
        得到每个mashup和api经过特征提取器或者平均池化得到的特征，可以直接用id索引，供构造instance的文本部分使用
        :param text_tag_recommend_model:
        :param mashup_num:
        :param api_num:
        :return:
        """
        mashup_texts_features, mashup_tag_features = self.get_mashup_text_tag_features (
            [m_id for m_id in range (mashup_num)])
        api_texts_features, api_tag_features = self.get_api_text_tag_features (
            [a_id for a_id in range (api_num)])
        api_text_features_path = os.path.join(self.model_dir,'api_text_features.dat')
        api_tag_features_path = os.path.join(self.model_dir,'api_tag_features.dat')
        save_2D_list(api_text_features_path, api_texts_features, 'w+')
        save_2D_list(api_tag_features_path, api_tag_features, 'w+')
        return mashup_texts_features,mashup_tag_features,api_texts_features,api_tag_features


# 在新的内容交互的基础上搭建新的完整模型
class gx_text_tag_continue_only_MLP_model (gx_text_tag_continue_model):

    def __init__(self,new_old,if_tag=True,if_tag_sim=True):

        super (gx_text_tag_continue_only_MLP_model, self).__init__ (new_old)
        self.if_tag = if_tag # 内容交互部分是否只使用text
        self.mhs = None
        self.mashup_sims_dict = {}  # (mashup_id,mashup_slt_apis_list：sims) 共用  或者mashup_id

        self.if_tag_sim = if_tag_sim # 相似度部分是否使用tag feature 计算的相似度
        self.HIN_dim = 2 if new_Para.param.if_mashup_sim_only else 6 # 默认使用tag计算相似度！！！

        # i_factors_matrix 按照全局id排序; UI矩阵也是按照api id从小到大排序
        # 而mashup不同，按照训练集中的内部索引大小
        self.num_feat = new_Para.param.num_feat
        # 加入一个padding用的虚拟的api的映射
        self.i_factors_matrix = np.zeros((meta_data.api_num+1,self.num_feat)) # 待改！！！传参和参数类，数据类之间的平衡
        for id,index in dataset.UV_obj.a_id2index.items():
            self.i_factors_matrix[id]=dataset.UV_obj.a_embeddings[index]

        self.m_id2index =  dataset.UV_obj.m_id2index
        self.m_index2id = {index: id for id, index in dataset.UV_obj.m_id2index.items ()}

        self.topK = new_Para.param.topK # 对模型有影响

        self.NI_handle_slt_apis_mode = new_Para.param.NI_handle_slt_apis_mode
        self.if_implict = new_Para.param.if_implict
        self.CF_self_1st_merge= new_Para.param.CF_self_1st_merge
        self.cf_unit_nums=new_Para.param.cf_unit_nums

        self.if_explict = new_Para.param.if_explict # 0是没有，1是跟所有api的共现次数向量；2是跟最近邻mashup调用过的api的共现次数；3是最近邻mashup是否调用过该api，50D
        self.co_unit_nums=new_Para.param.shadow_co_fc_unit_nums

        # 可组合性
        self.if_correlation = new_Para.param.if_correlation
        self.cor_fc_unit_nums = new_Para.param.cor_fc_unit_nums

        # self.api_id2covec,self.api_id2pop = meta_data.pd.get_api_co_vecs()

        self.predict_fc_unit_nums = new_Para.param.predict_fc_unit_nums

        self.model = None
        self.text_sim_dict =None
        self.tag_sim_dict = None

        self.simple_name='new_whole' if self.old_new=='new' else 'old_whole'

        if new_Para.param.no_CI:  # 整个模型只有NI,不使用CI!!!
            self.simple_name += '_noCI'
        self.no_CI = new_Para.param.no_CI

        if self.if_implict: # 隐式交互
            self.simple_name += '_implict_'
            if not self.NI_handle_slt_apis_mode:
                self.simple_name += 'noSlts_'
            else:
                self.simple_name += (self.NI_handle_slt_apis_mode+'_')
            self.simple_name += new_Para.param.mf_mode
            if not self.CF_self_1st_merge:
                self.simple_name += '_NoMLP_'

        if self.if_explict: # 显式交互
            self.simple_name += '_explict'
        self.simple_name += '_top{}'.format(self.topK)

        if self.no_CI and self.if_implict: # 只NI
            self.lr = new_Para.param.NI_learning_rate

        else:
            self.lr = 0.0003
        self.optimizer = Adam(self.lr)

        # 有关HIN_sim, NI模型和整个模型应该显示相似度设置！
        self.simple_name += '_HINSimParas_{}{}{}_'.format(*new_Para.param.HIN_sim_paras)
        self.simple_name += ('lr_{}'.format(self.lr))

        HIN_sim_name = 'if_mashup_sem:{} if_api_sem={} if_mashup_sim_only:{}'.format(*new_Para.param.HIN_sim_paras)
        text_tag = '_if_tag:{} if_tag_sim:{} _'.format(if_tag,if_tag_sim)
        ex_ = '_explict{}:{}'.format(self.if_explict, self.co_unit_nums).replace(',', ' ') if self.if_explict else ''
        im_ = '_implict:{}'.format(self.cf_unit_nums).replace(',', ' ') if self.if_implict else ''
        correlation = '_correlation:{}'.format(self.cor_fc_unit_nums).replace(',', ' ') if self.if_correlation else ''
        self.model_name += '_KNN_' + str(self.topK) + text_tag+ HIN_sim_name + ex_ + im_ + correlation

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        """
        self.model_name区分模型框架，返回的name用于记录在evalute中，区分不同的模型，架构
        :return:
        """
        return self.simple_name + self.model_name

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
        if self.if_tag: # 调用gx_text_tag_continue_model的方法，使用text和tag的feature
            self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features = \
                recommend_model.get_mashup_api_features(self.all_mashup_num, self.all_api_num)
            # api 需要增加一个全为0的，放在最后，id为api_num，用来对slt_apis填充
            self.api_tag_features = np.vstack((self.api_tag_features, np.zeros((1, self.word_embedding_dim))))

        else: # 调用gx_text_only_model的方法，
            self.mashup_texts_features, self.api_texts_features = recommend_model.get_mashup_api_text_features()

            if self.if_tag_sim: # 但是同时想使用tag sim，需要根据word embedding得到每个mashup和api的tag feature
                mashup_tag_features = []
                api_tag_features =[]

                if self.encoded_texts is None:
                    self.process_texts()  # 对所有文本编码，得到self.encoded_texts  因为没有使用word embedding layer层

                for m_id in range(self.num_users):
                    m_tags = self.encoded_texts.texts_in_index_nopadding[self.num_users+self.num_items+m_id]
                    m_tags_embeddings = np.array([self.wordindex2emb[word_index]  for word_index in m_tags])  # 2D
                    m_tags_features = np.average(m_tags_embeddings,axis=0)
                    mashup_tag_features.append(m_tags_features)
                for a_id in range(self.num_items):
                    a_tags = self.encoded_texts.texts_in_index_nopadding[2*self.num_users+self.num_items+a_id]
                    a_tags_embeddings = np.array([self.wordindex2emb[word_index]  for word_index in a_tags])  # 2D
                    a_tags_features = np.average(a_tags_embeddings,axis=0)
                    api_tag_features.append(a_tags_features)

                self.mashup_tag_features = np.array(mashup_tag_features)
                self.api_tag_features = np.array(api_tag_features)
                self.api_tag_features = np.vstack((self.api_tag_features, np.zeros((1, self.word_embedding_dim))))

        self.api_texts_features = np.vstack((self.api_texts_features, np.zeros((1, self.inception_fc_unit_nums[-1]))))
        self.features = (self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features)

    def set_embedding_weights(self,recommend_model):
        self.wordindex2emb = np.squeeze(recommend_model.embedding_layer.get_weights())  # !!! text_tag_model.get_layer ('embedding_layer')
        print(np.array(self.wordindex2emb).shape)
        print('after training the text_tag model, the embedding of padding 0:')
        print (self.wordindex2emb[0])
        path = os.path.join(recommend_model.model_dir,'word_embedding.txt')
        save_2D_list(path,self.wordindex2emb)
        print('the embedding of padding 0:')

    """
    def get_mashup_id2neighbors_HIN(self,mhs,m_id1, m_id2, mashup_slt_apis_list):
        # 基于HIN的思路，重新计算SBS的相似度，查找近邻
        # 但是文本相似度的计算还是基于之前的word-embedding的计算方法，加上提取的特征的相似度？
        pass

    """
    def build_m_i_matrix(self):
        # 根据训练数据构建历史的U-I矩阵,mashup按照内部索引顺序， api按全局id！！！ i_factors也应该与之对应！
        self.m_a_matrix=np.zeros((len(self.his_m_ids),self.all_api_num),dtype='int32')
        for index in range(len(dataset.crt_ds.train_mashup_id_list)):
            if  dataset.crt_ds.train_labels[index]==1:
                m_index=self.m_id2index[dataset.crt_ds.train_mashup_id_list[index]]
                self.m_a_matrix[m_index][dataset.crt_ds.train_api_id_list[index]]=1

    def prepare(self,recommend_model):
        # 为搭建模型做准备
        self.HIN_path = os.path.join(recommend_model.model_dir,'HIN_sims')
        self.model_dir = os.path.join(recommend_model.model_dir,self.simple_name)

        self.set_attribute()
        self.set_embedding_weights(recommend_model) # 复用func_only的embedding层权重  wordindex2emb
        self.set_mashup_api_features(recommend_model) # 复用func_only学到的mashup和api的text，tag特征
        self.build_m_i_matrix()

    def cpt_feature_sims_dict(self,mashup_features):
        # 得到每个mashup到历史mashup的text/tag特征的余弦相似度
        sim_dict={}
        for m_id in range(self.all_mashup_num):
            for his_m_id in self.his_m_ids:
                min_m_id=min(m_id,his_m_id)
                max_m_id =max(m_id,his_m_id)
                sim_dict[(min_m_id,max_m_id)]= cos_sim(mashup_features[min_m_id],mashup_features[max_m_id])
        return sim_dict

    def get_feature_sim(self,sim_dict,m_id1,m_id2):
        if m_id1==m_id2:
            return 0
        else:
            return sim_dict[(min(m_id1,m_id2),max(m_id1,m_id2))]

    def get_instances(self,mashup_id_list, api_id_list,mashup_slt_apis_list=None):
        if self.encoded_texts is None:
            self.process_texts()  # 对所有文本编码，得到self.encoded_texts  因为没有使用word embedding layer层

        # 每个待测mashup和历史mashup的HIN相似度（训练样本与自身的相似度为0
        if self.mhs is None:  # 一个训练和预测过程中使用的是同一个mhs对象
            features = self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features
            self.mhs = mashup_HIN_sims(self.wordindex2emb, self.encoded_texts, HIN_path=self.HIN_path,features=features)

        mashup_HIN_sims_instances = []
        for i in range(len(mashup_id_list)):  # 再改几种相似度计算方法
            # print('getting {}/{} instances HIN_sims...'.format(i, len(mashup_id_list)))
            m_id = mashup_id_list[i]
            if new_Para.param.need_slt_apis:
                _value = [self.mhs.get_mashup_HIN_sims(m_id, his_m_id, mashup_slt_apis_list[i])
                                                        if his_m_id != m_id else [0.0] * self.HIN_dim for his_m_id in self.his_m_ids]  # 2D  his_mashups*6
            else:
                _value = [self.mhs.get_mashup_HIN_sims(m_id, his_m_id)
                                                        if his_m_id != m_id else [0.0] * self.HIN_dim for his_m_id in self.his_m_ids]
            mashup_HIN_sims_instances.append(_value)

        # padding, the index of slt_item_num be mapped to zero in the embedding layer
        mashup_slt_apis_array=np.ones((len(mashup_id_list),new_Para.param.slt_item_num),dtype='int32')* self.all_api_num # default setting,api_num
        if self.old_new =='new':
            for i in range(len(mashup_slt_apis_list)):
                for j in range(len(mashup_slt_apis_list[i])):
                    mashup_slt_apis_array[i][j]= mashup_slt_apis_list[i][j]
        # print('get instances, done!')

        return np.array(mashup_id_list), np.array(api_id_list), np.array(mashup_HIN_sims_instances,dtype='float32'), mashup_slt_apis_array

    def get_instances_old(self,mashup_id_list, api_id_list,mashup_slt_apis_list=None):
        # features_tuple 根据gx_text_tag_continue_model.get_mashup_api_features()得到
        # his_m_ids 是每个历史mashup id
        # 获取每个mashup，api的特征不在这里做,在占空间;而是放在模型搭建时使用embedding得到.mashup_slt_apis_list最后要padding

        # 基于提取的text(tag特征)计算mashup和历史mashup的相似度
        if self.text_sim_dict is None:
            self.text_sim_dict=self.cpt_feature_sims_dict(self.mashup_texts_features)
        if self.if_tag_sim and self.tag_sim_dict is None:
            self.tag_sim_dict = self.cpt_feature_sims_dict (self.mashup_tag_features)
        # print('compute the cosine simility between texts and tags, done!')

        if self.encoded_texts is None:
            self.process_texts()  # 对所有文本编码，得到self.encoded_texts  因为没有使用word embedding layer层

        """
        if self.old_new =='new':

            # 每个待测mashup和历史mashup的相似度（训练样本与自身的相似度为0
            mhs = mashup_HIN_sims (self.wordindex2emb, self.encoded_texts)
            mashup_sims_dict={} # (mashup_id,mashup_slt_apis_list：sims) 共用
            mashup_HIN_sims_instances=[]
            
            mashup_text_fea_sims_instances = [] # 分别基于text和tag的feature衡量得到的sim
            mashup_tag_fea_sims_instances = []

            for i in range(len(mashup_id_list)):  # 再改几种相似度计算方法
                print('getting {}/{} instances HIN_sims...'.format(i,len(mashup_id_list)))

                m_id,slt_apis_tuple=mashup_id_list[i],tuple(mashup_slt_apis_list[i]) # tuple!! TypeError: unhashable type: 'list'
                _key=(m_id,slt_apis_tuple)
                if _key not in mashup_sims_dict.keys():
                    time_start = time.time()
                    # 该mashup到每个历史mashup的sim：二维  num*6  按照内部索引的顺序，和UI矩阵中mashup的索引相同
                    _value=[mhs.get_mashup_HIN_sims(m_id, his_m_id, mashup_slt_apis_list[i]) if his_m_id!= m_id else [0.0]*self.HIN_dim for his_m_id in self.his_m_ids] # 2D  his_mashups*6
                    mashup_sims_dict[_key]=_value
                    # print('first cpt,', _value)
                    time_end = time.time()
                    print('first cpt,cost time:{}'.format(time_end - time_start))
                else:
                    _value=mashup_sims_dict[_key]
                    # print('has cpted,')
                mashup_HIN_sims_instances.append(_value)

                mashup_text_fea_sims_instances.append([self.get_feature_sim(self.text_sim_dict,m_id,his_m_id) for his_m_id in self.his_m_ids])
                mashup_tag_fea_sims_instances.append ([self.get_feature_sim (self.tag_sim_dict, m_id, his_m_id) for his_m_id in self.his_m_ids])

            mhs.save_changes()
            with open(os.path.join(mhs.path, 'mashup_HIN_sims_instances'),'ab+') as f:
                pickle.dump(mashup_HIN_sims_instances, f)  # 先存储

            mashup_text_fea_sims_instances = np.expand_dims(np.array(mashup_text_fea_sims_instances),axis=2)
            mashup_tag_fea_sims_instances = np.expand_dims(np.array(mashup_tag_fea_sims_instances), axis=2)

            mashup_sims_instances=np.concatenate((np.array(mashup_HIN_sims_instances),mashup_text_fea_sims_instances,mashup_tag_fea_sims_instances),axis=2) # (?,his_mashups,8) 6-8-10
            with open(os.path.join(mhs.path, 'final_8sims_mashup_sims_instances'),'ab+') as f:
                pickle.dump(mashup_sims_instances, f)
        else:
        """

        # 相似度的计算非常重要，选择text tag feature和HIN_sim(会根据是否使用slt_apis变化)
        mashup_text_fea_sims_instances = [] # 使用text计算的相似度
        for m_id in mashup_id_list:
            mashup_text_fea_sims_instances.append(
                [self.get_feature_sim(self.text_sim_dict, m_id, his_m_id) for his_m_id in self.his_m_ids])
        mashup_text_fea_sims_instances = np.expand_dims(np.array(mashup_text_fea_sims_instances), axis=2)
        mashup_sims_instances = mashup_text_fea_sims_instances

        if self.if_tag_sim: # 使用tag计算的相似度
            mashup_tag_fea_sims_instances = []
            for m_id in mashup_id_list:
                mashup_tag_fea_sims_instances.append(
                    [self.get_feature_sim(self.tag_sim_dict, m_id, his_m_id) for his_m_id in self.his_m_ids])
            mashup_tag_fea_sims_instances = np.expand_dims(np.array(mashup_tag_fea_sims_instances), axis=2)

            mashup_sims_instances = np.concatenate((mashup_sims_instances, mashup_tag_fea_sims_instances),axis=2)  # (?,his_mashups,2)

        if self.if_HIN_sim:
            # 每个待测mashup和历史mashup的HIN相似度（训练样本与自身的相似度为0
            if self.mhs is None: # 一个训练和预测过程中使用的是同一个mhs对象
                self.mhs = mashup_HIN_sims(self.wordindex2emb, self.encoded_texts,HIN_path=self.HIN_path)
            mashup_HIN_sims_instances = []

            for i in range(len(mashup_id_list)):  # 再改几种相似度计算方法
                # print('getting {}/{} instances HIN_sims...'.format(i, len(mashup_id_list)))
                m_id = mashup_id_list[i]
                if new_Para.param.need_slt_apis:
                    slt_apis_tuple = tuple(mashup_slt_apis_list[i])  # tuple!! TypeError: unhashable type: 'list'
                    _key = (m_id, slt_apis_tuple)
                else:
                    _key = m_id
                if _key not in self.mashup_sims_dict.keys():
                    time_start = time.time()
                    # 该mashup到每个历史mashup的sim：二维  num*6  按照内部索引的顺序，和UI矩阵中mashup的索引相同
                    if new_Para.param.need_slt_apis:
                        _value = [self.mhs.get_mashup_HIN_sims(m_id, his_m_id, mashup_slt_apis_list[i]) if his_m_id != m_id else [0.0] * self.HIN_dim for his_m_id in self.his_m_ids]  # 2D  his_mashups*6
                    else:
                        _value = [self.mhs.get_mashup_HIN_sims(m_id, his_m_id) if his_m_id != m_id else [0.0] * self.HIN_dim for his_m_id in self.his_m_ids]
                    self.mashup_sims_dict[_key] = _value
                    # print('first cpt,', _value)
                    time_end = time.time()
                    print('first cpt,cost time:{}'.format(time_end - time_start))
                else:
                    _value = self.mashup_sims_dict[_key]
                    # print('has cpted,')
                mashup_HIN_sims_instances.append(_value)
            mashup_sims_instances = np.concatenate((mashup_sims_instances, np.array(mashup_HIN_sims_instances)), axis=2)

        # with open(os.path.join(dataset.crt_ds.root_path, 'final_{}sims_mashup_sims_instances.sim'.format(self.total_sim_dim)), 'ab+') as f:
        #     pickle.dump(mashup_sims_instances, f)
        # print('compute sims, done!')

        # padding, the index of slt_item_num be mapped to zero in the embedding layer
        mashup_slt_apis_array=np.ones((len(mashup_id_list),new_Para.param.slt_item_num),dtype='int32')* self.all_api_num # default setting,api_num
        if self.old_new =='new':
            for i in range(len(mashup_slt_apis_list)):
                for j in range(len(mashup_slt_apis_list[i])):
                    mashup_slt_apis_array[i][j]= mashup_slt_apis_list[i][j]
        # print('get instances, done!')

        return np.array(mashup_id_list), np.array(api_id_list), np.array(mashup_sims_instances,dtype='float32'), mashup_slt_apis_array

    def save_HIN_sim(self): # 一个epoch内一直更新，多个epoch间不用更新,第一个epoch之后保存该对象
        self.mhs.save_changes()

    def get_model(self,_model):
        # set the embedding value for the  padding value
        if self.model is None:
            mashup_id_input = Input (shape=(1,), dtype='int32', name='mashup_id_input')
            api_id_input = Input (shape=(1,), dtype='int32', name='api_id_input')

            m_sims_input = Input(shape=(len(self.his_m_ids),self.HIN_dim,), dtype='float32', name='mashup_sims_input')
            print('m_sims_input', m_sims_input)
            m_sims = Dense (1, activation='linear', use_bias=False, kernel_initializer='uniform', name="m_sims") (m_sims_input)  # (?,his_num,1)
            print('m_sims', m_sims)

            # 为了sim_lam好写，必须要有，instances可以随意设计输入值，占位
            slt_apis_input = Input (shape=(new_Para.param.slt_item_num,), dtype='int32', name='slt_apis_input')  # used for co-invoke 节省内存！
            print('slt_apis_input', slt_apis_input)

            # get the functional feature input, using the embedding layer instead of packing in the instance, to save memory
            def get_func_features_input():
                mashup_text_feature_embedding_layer = Embedding(self.all_mashup_num,self.inception_fc_unit_nums[-1],
                                                 embeddings_initializer=Constant(self.mashup_texts_features),
                                                 mask_zero=False,
                                                 trainable=False, name = 'mashup_text_feature_embedding_layer') # input_length=new_Para.param.slt_item_num,

                api_text_feature_embedding_layer = Embedding(self.all_api_num+1,self.inception_fc_unit_nums[-1],
                                                 embeddings_initializer=Constant(self.api_texts_features),
                                                 mask_zero=False,
                                                 trainable=False, name = 'api_text_feature_embedding_layer') # input_length=new_Para.param.slt_item_num,

                if self.if_tag:
                    mashup_tag_feature_embedding_layer = Embedding(self.all_mashup_num, self.word_embedding_dim,
                                                                   embeddings_initializer=Constant(self.mashup_tag_features),
                                                                   mask_zero=False,
                                                                   trainable=False, name = 'mashup_tag_feature_embedding_layer') # input_length=new_Para.param.slt_item_num,

                    api_tag_feature_embedding_layer = Embedding(self.all_api_num + 1, self.word_embedding_dim,
                                                                embeddings_initializer=Constant(self.api_tag_features),
                                                                mask_zero=False,
                                                                trainable=False, name = 'api_tag_feature_embedding_layer') # input_length=new_Para.param.slt_item_num,

                # (?, embedding)
                m_text_features = mashup_text_feature_embedding_layer(mashup_id_input) # shape=(?, 1, 50), dtype=float32
                a_text_features = api_text_feature_embedding_layer(api_id_input)
                m_text_features = Lambda(lambda x: tf.squeeze(x, axis=1))(m_text_features) # (None, 1, 50)->...
                a_text_features = Lambda(lambda x: tf.squeeze(x, axis=1))(a_text_features)

                if self.if_tag:
                    m_tag_features = mashup_tag_feature_embedding_layer(mashup_id_input)
                    a_tag_features = api_tag_feature_embedding_layer(api_id_input)
                    m_tag_features = Lambda(lambda x: tf.squeeze(x, axis=1))(m_tag_features)
                    a_tag_features = Lambda(lambda x: tf.squeeze(x, axis=1))(a_tag_features)

                # print('m_text_features',m_text_features)
                # print('api_id_input', api_id_input)

                if self.old_new=='new': # function部分是否使用已选择过的api
                    slt_apis_text_features = api_text_feature_embedding_layer(slt_apis_input)  # shape=(?, 3, 50), dtype=float32
                    print('slt_apis_text_features', slt_apis_text_features)

                    # (?,new_Para.param.slt_item_num,embedding) ->(?,new_Para.param.slt_item_num*embedding)
                    # K.reshpe() 函数不能直接用！！！！！！使用lambda 封装tf函数的方法可用！！！或者使用Keras的层
                    slt_apis_text_features= Lambda(lambda x:tf.reshape(x,[-1,new_Para.param.slt_item_num*self.inception_fc_unit_nums[-1]]))([slt_apis_text_features])
                    print('reshaped slt_apis_text_features', slt_apis_text_features)

                    if self.if_tag: # 使用文本和tag
                        slt_apis_tag_features = api_tag_feature_embedding_layer(slt_apis_input)
                        slt_apis_tag_features = Lambda(lambda x: tf.reshape(x, [-1, new_Para.param.slt_item_num * self.word_embedding_dim]))([slt_apis_tag_features])
                        print('reshaped slt_apis_tag_features', slt_apis_tag_features)

                        func_features_input = Concatenate()([m_text_features,m_tag_features,a_text_features,a_tag_features,slt_apis_text_features,slt_apis_tag_features])
                    else: # 只使用文本
                        func_features_input = Concatenate()([m_text_features, a_text_features, slt_apis_text_features])

                else:
                    if self.if_tag:  # 使用文本和tag notice the order!!!
                        func_features_input = Concatenate()([m_text_features,a_text_features,m_tag_features,a_tag_features])
                    else:
                        func_features_input = Concatenate()([m_text_features, a_text_features])

                return func_features_input

            func_features_input = get_func_features_input()

            u_factors = K.variable (dataset.UV_obj.m_embeddings, dtype='float32')  # 外部调用包，传入  存储
            i_factors = K.variable (self.i_factors_matrix, dtype='float32')  # 训练集中未出现过的api当作是新的api，factor为0
            m_a_matrix = K.variable (self.m_a_matrix, dtype='int32')

            # 之前的做法中，内部各种处理都是K，但没有把map_fn用lambda层嵌套，不行！！！K.map_fn()不能完成任务
            # lambda层封装的函数内部，可以随意使用tf的函数，最后lambda层会统一转化！
            # lambda层的输入输出都是list！！ map_fn() 输出与输入不同数目或者类型时，要声明！！
            def sim_lam(paras):
                _m_sims_input = paras[0]
                _slt_apis_input = paras[1]
                _api_id_input = paras[2]

                def fn(elements):
                    a_m_sims = K.squeeze (elements[0],axis=1)  # 默认消除维度为1的-> (his_num,)
                    a_slt_apis = K.cast (elements[1], tf.int32)  # 已选择的api: (new_Para.param.slt_item_num,)
                    a_api_id = K.cast (K.squeeze (elements[2],axis=0), tf.int32)  # 待测api id (1,)->()

                    max_sims, max_indexes = tf.nn.top_k (a_m_sims, self.topK)  # (50,),  是历史mashup的内部索引
                    max_sims = max_sims / K.sum (max_sims)  # 归一化
                    max_sims = K.reshape (max_sims, (self.topK, 1))

                    if self.if_implict:
                        print('u_factors:', u_factors)
                        neighbor_m_cf_feas = K.gather(u_factors, max_indexes)  # 最近邻的cf feature
                        print('neighbor_m_cf_feas:',neighbor_m_cf_feas) # (50, 25)
                        m_cf_feas = K.sum (max_sims * neighbor_m_cf_feas, axis=0)  # *:(50,25)->(25,)  该mquery的cf feature
                        print('m_cf_feas:', m_cf_feas)  # (25,)
                        a_cf_feas = i_factors[a_api_id]
                        print('a_cf_feas:', a_cf_feas) # shape=(25,)
                    else: # 相当于占位符，无实际作用，随意!!!
                        m_cf_feas = i_factors[a_api_id]
                        a_cf_feas = i_factors[a_api_id]

                    m_cf_feas = Lambda(lambda x: tf.cast(x, tf.float32))(m_cf_feas)
                    a_cf_feas = Lambda(lambda x: tf.cast(x, tf.float32))(a_cf_feas)

                    # co_invoke 计算量不大，无论如何都输出
                    column_vec = m_a_matrix[:, a_api_id]  # slide:-> (?,1) 最好是(his_m_ids_num,1)  squeeze:-> (?)
                    print('column_vec:', column_vec)  # shape=(1385,), dtype=int32
                    co_vec = K.gather(column_vec, max_indexes)  # neighbor 是否调用过该api: (50,) 0-1向量
                    print('co_vec after gather:', co_vec)  # shape=(50,), dtype=int32
                    co_vec = K.reshape(co_vec, (self.topK,))

                    if self.if_correlation:
                        # correlation,依赖于显式的co_vec
                        co_vec_rs = K.reshape(co_vec, (50,1))
                        row_vecs = K.gather (m_a_matrix, max_indexes)  # (50,api_num)
                        row_vecs = co_vec_rs * row_vecs  # 自动广播，邻居是否调用过待测api，没有的话行为0
                        print('row_vecs :', row_vecs) # shape=(50, 728), dtype=int32)

                        def one_hot(a_slt_apis):
                            # one_hot 不对padding的最大index 编码！
                            print('a_slt_apis before:',a_slt_apis)
                            a_slt_apis= tf.cast(a_slt_apis,'int32')
                            all_onehot_tensors = tf.one_hot(a_slt_apis, self.all_api_num, 1, 0) # 所有已选择的apis  one-hot化 2D
                            print('all_onehot_tensors:',all_onehot_tensors)
                            sum_onehot_tensors = tf.reduce_sum(all_onehot_tensors,axis=0,keep_dims=True) # 全部相加，得到一个选择api上值为1的向量 (1,728)?
                            print('sum_onehot_tensors:', sum_onehot_tensors) # (1, 728)
                            return sum_onehot_tensors

                        trans_a_slt_apis = Lambda(one_hot)(a_slt_apis) # 得到multi-one-hot的向量，表示用户已选择的api
                        # 每一维是某个近邻中，跟待测api一起出现的 已选择过的api的比例
                        correlation_vec = K.sum (row_vecs * trans_a_slt_apis, axis=1) / K.sum (trans_a_slt_apis)
                        print('correlation_vec:',correlation_vec) # (50,)
                    else:
                        correlation_vec = co_vec # 占位，随意
                    correlation_vec = Lambda(lambda x: tf.cast(x, tf.float32))(correlation_vec)
                    co_vec = Lambda(lambda x: tf.cast(x, tf.float32))(co_vec)

                    return  m_cf_feas,a_cf_feas, co_vec,correlation_vec # map_fn的返回值要跟输入相同类型！！！！tuple！！！
                return list(K.map_fn(fn, (_m_sims_input, _slt_apis_input, _api_id_input),dtype=(tf.float32,tf.float32,tf.float32,tf.float32)))

            # 旧模型或者不需要correlation_vecs时它无意义，占位而已
            m_cf_feas, a_cf_feas,co_vecs,correlation_vecs = Lambda(sim_lam)([m_sims, slt_apis_input, api_id_input])

            if not self.no_CI: # 使用CI部分
                predict_vector = func_features_input
                print('func_features_input',func_features_input)
                for index, unit_num in enumerate(new_Para.param.content_fc_unit_nums):
                    predict_vector = Dense(unit_num, activation='relu',
                                           kernel_regularizer=l2(new_Para.param.l2_reg), name='content_dense_{}'.format(index))(predict_vector)

            if self.if_implict:
                if self.NI_handle_slt_apis_mode:
                    api_implict_embedding_layer = Embedding(self.all_api_num+1,
                                                                    self.num_feat,
                                                                    embeddings_initializer=Constant(self.i_factors_matrix),
                                                                    mask_zero=False,
                                                                    trainable=False,
                                                                    name='api_implict_embedding_layer')  # input_length=new_Para.param.slt_item_num,

                    slt_api_implict_embeddings = api_implict_embedding_layer(slt_apis_input) # (None,3,25)

                    if self.NI_handle_slt_apis_mode=='attention':
                        slt_api_implict_embeddings = attention(a_cf_feas,slt_api_implict_embeddings,slt_api_implict_embeddings)
                    elif self.NI_handle_slt_apis_mode=='full_concate':
                        slt_api_implict_embeddings = Reshape((self.num_feat * new_Para.param.slt_item_num))(slt_api_implict_embeddings)  # (None,75)

                if self.CF_self_1st_merge: # cf特征先使用MLP处理
                    predict_vector2 = Concatenate()([m_cf_feas, a_cf_feas])
                    if self.NI_handle_slt_apis_mode: # 需要已选择的api的隐式向量
                        predict_vector2 = Concatenate()([predict_vector2, slt_api_implict_embeddings])

                    for unit_num in self.cf_unit_nums:
                        predict_vector2 = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector2)

                else: # cf特征使用element-wise 乘法处理
                    predict_vector2 = Multiply()([m_cf_feas, a_cf_feas])

                if not self.no_CI:  # 使用CI部分
                    predict_vector = Concatenate()([predict_vector, predict_vector2])
                else:
                    predict_vector = predict_vector2

            if self.if_explict:  # 显式历史交互
                predict_vector3 = Dense (self.co_unit_nums[0], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (co_vecs)
                for unit_num in self.co_unit_nums[1:]:
                    predict_vector3 = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector3)
                predict_vector = Concatenate()([predict_vector, predict_vector3])

            if self.old_new== 'new' and self.if_correlation:
                predict_vector4 = Dense (self.cor_fc_unit_nums[0], activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (correlation_vecs)  # 暂时对co_vecs和correlation_vecs使用相同结构的MLP
                for unit_num in self.cor_fc_unit_nums[1:]:
                    predict_vector4 = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector4)
                predict_vector = Concatenate()([predict_vector, predict_vector4])

            if self.CF_self_1st_merge and self.no_CI: # 只使用NI部分预测，不需要顶层的NLP，->...->50->1
                pass
            else:
                for unit_num in self.predict_fc_unit_nums:
                    predict_vector = Dense (unit_num, activation='relu', kernel_regularizer = l2(new_Para.param.l2_reg)) (predict_vector)

            predict_vector = Dropout (0.5) (predict_vector)
            predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
                predict_vector)

            self.model = Model (inputs=[mashup_id_input,api_id_input, m_sims_input, slt_apis_input],
                                outputs=[predict_result])
            print(self.get_name())

            for layer in self.model.layers:
                print(layer.name)

            print('build model,done!')

            if not self.no_CI:
                denseWlist =[]
                for index, unit_num in enumerate(new_Para.param.content_fc_unit_nums[:-1]):  # 需要名字！！！
                    denseWlist.append(_model.get_layer('content_dense_{}'.format(index)).get_weights())
                denseWlist.append(_model.get_layer('text_tag_feature_extracter').get_weights())

                for index in range(len(new_Para.param.content_fc_unit_nums)):
                    self.model.get_layer('content_dense_{}'.format(index)).set_weights(denseWlist[index])
                """
                w_dense2 = _model.get_layer('dense_1').get_weights()
                w_dense3 = _model.get_layer('dense_2').get_weights()
                w_dense4 = _model.get_layer('text_tag_feature_extracter').get_weights()
    
                self.model.get_layer('cf1').set_weights(w_dense2)
                self.model.get_layer('cf2').set_weights(w_dense3)
                self.model.get_layer('cf3').set_weights(w_dense4)
            """
        return self.model

    def save_sim_weight(self):
        # 相似度层的权重
        sim_weight = self.model.get_layer("m_sims").get_weights()
        print('sim_weight:',sim_weight)
        root = dataset.crt_ds.model_path.format(self.get_simple_name())
        sim_weight_path = os.path.join(root, 'sim_weight')
        np.save(sim_weight_path,sim_weight)
        print('save sim_weight,done!')