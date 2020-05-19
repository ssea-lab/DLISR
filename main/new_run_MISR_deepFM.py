import pickle
import sys
sys.path.append("..")

import numpy as np
from tensorflow.python.keras.utils import to_categorical
from main.evalute import evalute, summary
from main.new_para_setting import new_Para
from deepCTR.models import simple_DeepFM
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

def getNum_testData(test_data):
    # 获得测试样例的个数
    test_mashup_id_list, test_api_id_list = test_data[:2]
    num = 0
    for i in range(len(test_mashup_id_list)):
        num += len(test_api_id_list[i])
    return num

def transfer_testData2(test_data):
    test_mashup_id_list, test_api_id_list,test_mashup_slt_apis_list = test_data[:-1]
    test_mashup_id_list_, test_api_id_list_,test_mashup_slt_apis_list_ = [], [], []
    num = 0
    for i in range(len(test_mashup_id_list)):
        for j in range(len(test_api_id_list[i])):
            test_mashup_id_list_.append(test_mashup_id_list[i][j])
            test_api_id_list_.append(test_api_id_list[i][j])
            test_mashup_slt_apis_list_.append(test_mashup_slt_apis_list[i])
            num+=1
    return test_mashup_id_list_, test_api_id_list_,test_mashup_slt_apis_list_


def data_generator(data, mashup_texts_features, mashup_tag_features, api_texts_features, api_tag_features, mashup_NI_features, api_NI_features, bs,all_api_num, mode="train"):
    i = 0
    if mode=="train":
        mashup_id_list, api_id_list,mashup_slt_apis_list = data[:-1]
        label_list = data[-1]
        labels = to_categorical(label_list, num_classes=2)  # 针对softmax,numpy
    elif mode=="test":
        mashup_id_list, api_id_list, mashup_slt_apis_list = transfer_testData2(data)

    # 填充
    mashup_slt_apis_array = np.ones((len(mashup_id_list), new_Para.param.slt_item_num),dtype='int32') * all_api_num  # default setting,api_num
    for i in range(len(mashup_slt_apis_list)):
        for j in range(len(mashup_slt_apis_list[i])):
            mashup_slt_apis_array[i][j] = mashup_slt_apis_list[i][j]

    num_instances = len(mashup_id_list)
    while True: # train时生成batch的无限循环，test一轮即可
        m_text_feas,m_tag_feas,a_text_feas,a_tag_feas = [],[],[],[]
        m_NI_feas,a_NI_feas = [],[]
        slt_apis_text_feas,slt_apis_tag_feas,slt_apis_NI_feas = [],[],[]
        if mode == "train":
            return_labels = []

        while len(m_text_feas)< bs:
            i = i % (num_instances)
            m_id = mashup_id_list[i]
            a_id = api_id_list[i]
            slt_apis = mashup_slt_apis_list[i]
            padded_slt_apis = mashup_slt_apis_array[i]

            m_text_feas.append(mashup_texts_features[m_id])
            m_tag_feas.append(mashup_tag_features[m_id])
            a_text_feas.append(api_texts_features[a_id])
            a_tag_feas.append(api_tag_features[a_id])
            m_NI_feas.append(mashup_NI_features[(m_id, tuple(slt_apis))])
            a_NI_feas.append(api_NI_features[a_id])

            slt_apis_text_feas.append([api_texts_features[api_id] for api_id in padded_slt_apis])
            slt_apis_tag_feas.append([api_tag_features[api_id] for api_id in padded_slt_apis])
            slt_apis_NI_feas.append([api_NI_features[api_id] for api_id in padded_slt_apis])

            if mode == "train":
                return_labels.append(labels[i])

            i += 1
            if mode == "test" and i==num_instances:
                break

        features = [np.array(m_text_feas),np.array(m_tag_feas),np.array(a_text_feas),np.array(a_tag_feas),
                np.array(m_NI_feas), np.array(a_NI_feas),np.array(slt_apis_text_feas), np.array(slt_apis_tag_feas), np.array(slt_apis_NI_feas)]

        if mode == "train":
            yield (features,np.array(return_labels))
        elif mode == "test":
            yield (features)


def run_new_deepFM(CI_feas,NI_feas,train_data,test_data,all_api_num,epoch_num=10):
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)
    # graph = tf.get_default_graph()
    # set_session(session)

    model = simple_DeepFM(CI_feature_num=4, NI_feature_num=2, CI_feature_dim=50, NI_feature_dim=25,final_feature_dim=32,task='binary',
                          use_fm=True,l2_reg_linear=0,dnn_hidden_units=[])
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
    print('bulid simple_DeepFM,done!')
    
    batch_size = 32
    len_train = len(train_data[0])

    mashup_texts_features, mashup_tag_features, api_texts_features, api_tag_features = CI_feas
    mashup_NI_features,api_NI_features = NI_feas

    features = [mashup_texts_features, mashup_tag_features, api_texts_features, api_tag_features,mashup_NI_features,api_NI_features]
    train_generator = data_generator(train_data,*features, bs=batch_size,all_api_num = all_api_num,mode="train")
    print('genarate train_generator ,done!')

    # 每训练一次就测试一次
    num_test_instances = getNum_testData(test_data)
    for i in range(epoch_num):
        history = model.fit_generator(train_generator, steps_per_epoch=len_train // batch_size, epochs=1, verbose=2)
        test_generator = data_generator(test_data,*features,bs=batch_size,all_api_num = all_api_num,mode="test")
        print('genarate test_generator,done!')
        predictions = model.predict_generator(test_generator, steps=num_test_instances // batch_size+1)[:, 1]
        print(predictions.shape)

        reshaped_predictions = []
        # 评价
        test_api_id_list, grounds = test_data[1], test_data[-1]
        index = 0
        for test_api_ids in test_api_id_list:
            size = len(test_api_ids)  # 当前mashup下的候选api的数目
            reshaped_predictions.append(predictions[index:index + size]) # min(index + size,len(predictions))
            index += size
        print(index)
        evaluate_result = evalute(test_api_id_list, reshaped_predictions, grounds, new_Para.param.topKs)  # 评价
        summary(new_Para.param.evaluate_path, 'deepFM_epoch_{}'.format(i), evaluate_result, new_Para.param.topKs)  #
