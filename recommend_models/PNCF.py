import sys
sys.path.append('..')

from keras import Input, Model
from keras.layers import Multiply, Concatenate, Dense, Dropout
from keras.regularizers import l2
from main.new_para_setting import new_Para
from recommend_models.baseline import get_default_gd
from recommend_models.recommend_Model import gx_model
from main.dataset import dataset
import os
import numpy as np

class PNCF_model(gx_model):
    def __init__(self,mlp_mode='GMF',concate_mode='multiply',text_mode='HDP' ,LDA_topic_num=None):
        # GMF：先元素乘，然后直接dense1，但是这里没有embedding，直接这么做效果可能很差；
        # MLP：先拼接(或元素乘)，再MLP
        # concate_mode: 'multiply','concate'
        super(PNCF_model, self).__init__()
        self.mlp_mode = mlp_mode
        self.concate_mode = concate_mode
        self.text_mode = text_mode
        self.LDA_topic_num = LDA_topic_num
        self.simple_name = 'PNCF_model_{}_{}_{}_{}'.format(mlp_mode,concate_mode,text_mode,LDA_topic_num)
        self.initilize()

    def get_simple_name(self):
        return self.simple_name

    def get_name(self):
        # self.model_name 不变
        name = self.simple_name  # + self.model_name
        return name

    def initilize(self):
        root = os.path.join(dataset.crt_ds.root_path ,'baselines')
        if not os.path.exists(root):
            os.makedirs(root)
        mashup_feature_path =os.path.join(root, 'mashup_{}.txt'.format(self.text_mode)) # ...
        api_feature_path = os.path.join(root, 'api_{}.txt'.format(self.text_mode))

        # 获取mashup_hdp_features,api_hdp_features
        if not os.path.exists(api_feature_path):
            gd =get_default_gd()
            self._mashup_features ,self._api_features =gd.model_pcs(self.text_mode,self.LDA_topic_num)
            np.savetxt(mashup_feature_path ,self._mashup_features)
            np.savetxt(api_feature_path, self._api_features)
        else:
            self._mashup_features =np.loadtxt(mashup_feature_path)
            self._api_features =np.loadtxt(api_feature_path)

    def get_model(self):
        if self.model is None:
            feature_size = self._mashup_features.shape[1]
            mashup_fea_input = Input (shape=(feature_size,), dtype='float32', name='mashup_fea_input')
            api_fea_input = Input (shape=(feature_size,), dtype='float32', name='api_fea_input')

            if self.concate_mode == 'multiply':
                x = Multiply()([mashup_fea_input, api_fea_input])
            elif self.concate_mode == 'concate':
                x = Concatenate(name='fea_concatenate')([mashup_fea_input, api_fea_input])
            else:
                print('wrong concate_mode!')
                sys.exit(1)

            if self.mlp_mode == 'MLP':
                for index, unit_num in enumerate(self.content_fc_unit_nums):  # 需要名字！！！
                    x = Dense(unit_num, activation='relu', kernel_regularizer=l2(new_Para.param.l2_reg),name='content_dense_{}'.format(index))(x)
            # GMF这里不用处理，上面的元素乘就够了

            x = Dropout(0.5)(x)
            predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (x)
            #

            self.model = Model(inputs=[mashup_fea_input, api_fea_input], outputs=[predict_result])

        return self.model

    def get_instances(self, mashup_id_instances, api_id_instances):
        examples = (
            np.array([self._mashup_features[mashup_id] for mashup_id in mashup_id_instances]),
            np.array([self._api_features[api_id] for api_id in api_id_instances]),
        )
        return examples