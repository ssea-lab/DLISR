import os
import pickle
from tensorflow.python.keras.models import Model

class midMLP_feature_obj(object):
    # 得到并存储CI,NI模型的中间层结果，供topMLP复用
    def __init__(self,recommend_model,model,if_explict= False):
        self.midMLP_feature_model = None
        self.midMLP_feature_flag = False # 某一次新求的话，最后训练和测试结束后更新一次
        self.midMLP_fea_path = os.path.join(recommend_model.model_dir,'midMLP_feature.dat')
        self.input2fea = None # 存储输入到中间输出的字典,训练和测试集全部的
        self.recommend_model = recommend_model
        self.model = model
        self.explict = if_explict # NI显式模型，旧场景

    def get_midMLP_feature(self, layer_name, mashup_id_instances, api_id_instances, slt_api_ids_instances=None,test_phase_flag=True):
        """
        根据输入得到最上层MLP中间结果，分批，训练和batch测试时多次调用
        layer_name:
        CI: 'text_tag_feature_extracter' ;
        隐式NI: 'implict_dense_{}'.format(len()-1)
        显式NI: 'explict'/'cor'_dense_{}'.format(len()-1)
        :return:
        """
        if self.input2fea is None:  # 第一次用
            if os.path.exists(self.midMLP_fea_path):
                with open(self.midMLP_fea_path, 'rb') as f:
                    self.input2fea = pickle.load(f)
                    print('load existed midMLP_fea_file!')
            else:
                print('no existed midMLP_fea_file!')
                self.input2fea = {}  # 输入到feature，不用numpy因为测试样例格式特殊
                self.midMLP_feature_flag = True

        num = len(mashup_id_instances)

        if slt_api_ids_instances is not None:
            # 成批的(训练集或一个batch的测试集)，一个不在的话全部没算过
            if (mashup_id_instances[0], api_id_instances[0], tuple(slt_api_ids_instances[0])) not in self.input2fea:
                self.midMLP_feature_flag = True
                if self.midMLP_feature_model is None:
                    self.midMLP_feature_model = Model(inputs=[*self.model.inputs],
                                                      outputs=[self.model.get_layer(layer_name).output])

                instances = self.recommend_model.get_instances(mashup_id_instances, api_id_instances, slt_api_ids_instances,test_phase_flag=test_phase_flag)
                midMLP_features = self.midMLP_feature_model.predict([*instances], verbose=0)  # 现求的二维numpy
                # 存到dict中
                for index in range(num):
                    key = (mashup_id_instances[index], api_id_instances[index], tuple(slt_api_ids_instances[index]))
                    self.input2fea[key] = midMLP_features[index]
            else:
                midMLP_features = [self.input2fea[(mashup_id_instances[index], api_id_instances[index], tuple(slt_api_ids_instances[index]))]
                                   for index in range(num)]
        else:
            midMLP_features = []
            for index in range(num):
                key = (mashup_id_instances[index], api_id_instances[index])
                if key not in self.input2fea:
                    self.midMLP_feature_flag = True
                    if self.midMLP_feature_model is None:
                        if not self.explict:
                            self.midMLP_feature_model = Model(inputs=[*self.model.inputs],outputs=[self.model.get_layer(layer_name).output])
                        else:
                            self.midMLP_feature_model = Model(input=self.model.input,output=self.model.get_layer(layer_name).output)
                    a_instance_list = self.recommend_model.get_instances([key[0]], [key[1]],test_phase_flag=test_phase_flag)

                    if not self.explict:
                        a_midMLP_feature = self.midMLP_feature_model.predict([*a_instance_list], verbose=0)[0]  # 现求的二维numpy
                    else:
                        a_midMLP_feature = self.midMLP_feature_model.predict(a_instance_list, verbose=0)

                    self.input2fea[key] = a_midMLP_feature
                else:
                    a_midMLP_feature= self.input2fea[key]
                midMLP_features.append(a_midMLP_feature)
        return midMLP_features

    def save_sth(self): # 结束之后存储中间feature
        if self.midMLP_feature_flag:
            with open(self.midMLP_fea_path, 'wb+') as f:
                pickle.dump(self.input2fea, f)
            print('save midMLP_fea_file,done!')