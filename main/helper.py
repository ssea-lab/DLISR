import sys

from main.dataset import dataset

sys.path.append("..")
import numpy as np


# 已经求过特征和text_tag_model/已经求过特征和text_tag_CF_model参数，利用中间结果
def get_iniFeaturesAndParas(x_model, para_mode):
    with open(Para.best_epoch_path.format(para_mode), 'r') as f:
        best_epoch=int(f.readline())
    para_path= Para.model_para_path.format(para_mode, best_epoch)

    x_model.load_weights(para_path)

    ini_mashup_text_feature=np.loadtxt(Para.text_features_path.format(para_mode))
    ini_mashup_tag_feature=np.loadtxt(Para.tag_features_path.format(para_mode))
    ini_features_array = np.hstack((ini_mashup_text_feature, ini_mashup_tag_feature))  # 整合的mashup的特征
    print('read iniFeaturesAndParas,done!')
    return ini_features_array, x_model


def load_trained_model(recommend_model, model):
    """
    只需要载入并返回训练好的模型即可
    :param recommend_model:
    :param para_mode:
    :return:
    """
    with open(dataset.crt_ds.new_best_epoch_path.format(recommend_model.model_dir), 'r') as f:
        best_epoch=int(f.readline())
    para_path= dataset.crt_ds.new_model_para_path.format(recommend_model.model_dir, best_epoch)
    model.load_weights(para_path)
    print('load model:{},done!'.format(recommend_model.simple_name))
    return model
