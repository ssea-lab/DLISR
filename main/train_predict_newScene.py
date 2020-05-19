# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("..")

from main.dataset import dataset,meta_data
from main.new_para_setting import new_Para, new_Para_
from main.new_split_dataset import split_dataset_for_newScene_New_KCV, split_dataset_for_oldScene_KCV
from main.run_models import test_simModes, bl_PasRec, DINRec, bl_DHSR_new, baselines, bl_IsRec_best

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def tst_kcv_paraObj(old_new = 'new', if_few=False): # 新模型还是旧模型
    # 先一次性生成完毕，否则即时生成再训练，容易出问题
    def get_dataset_generator():
        if new_Para.param.data_mode == 'newScene':
            dataset_generator = split_dataset_for_newScene_New_KCV(*new_Para.param.split_newScene_dataset_settings)
        elif new_Para.param.data_mode == 'oldScene':
            dataset_generator = split_dataset_for_oldScene_KCV(*new_Para.param.split_oldScene_dataset_settings)
        return dataset_generator

    i = 0
    for a_dataset in get_dataset_generator():
        print('getting the {}th kcv...'.format(i))
        i+=1

    text_recommend_model = None
    text_model = None
    # KCV形式
    end_index=4 #
    start_index=0
    index = 0
    for a_dataset in get_dataset_generator():
        print('kcv:{}'.format(index))
        if index<start_index:
            index += 1
            continue
        if index>end_index:
            return

        dataset.set_current_dataset(a_dataset)
        index+=1

        # cotrain_CINI()
        baselines(a_dataset)
        # bl_DHSR()
        # bl_DHSR_new(a_dataset)
        # text_tag()
        # CI_NI_fineTuning()

        # NI_online() # 最新的模型
        # co_trainCINI()
        # test_PNCF_doubleTower_OR_DIN()

        # bl_IsRec()
        # bl_IsRec_best(a_dataset)
        # bl_PasRec(a_dataset)
        # deepFM()
        # newDeepFM() # 效果很差
        # test()
        # a_dataset.transfer() # 新数据对不加已选的模型的简化, 用在CI和NI的need_slt_apis = False中
        # test_simModes(a_dataset,old_new,if_few = if_few)
        # DINRec(a_dataset,old_new)


def get_new_HIN_paras(HIN_mode):
    if HIN_mode == 'DeepCos':
        new_HIN_paras = [None, None, 'Deep', None, 'Deep']
    elif HIN_mode == 'DeepText_MetaPathTag':
        new_HIN_paras = ['MetaPath', None, None, None, 'Deep']  # 文本用deep;  tag用非语义
    elif HIN_mode == 'HDPText_DeepTag':
        new_HIN_paras = [None, None, 'Deep', None, 'HDP']  # 文本用HDP;  tag用deep
    elif HIN_mode == 'HDPText_MetaPathTag':
        new_HIN_paras = ['MetaPath', None, None, None, 'HDP']
    elif HIN_mode == 'EmbMaxText_DeepTag':
        new_HIN_paras = [None, None, 'Deep', 'EmbMax', None]  # 文本用HDP;  tag用deep
    elif HIN_mode == 'HDPText_EmbMaxTag':
        new_HIN_paras = ['EmbMax', None, None, None, 'HDP']  # 文本用HDP/TF_IDF;  tag用非语义
    elif HIN_mode == 'TFIDFText_MetaPathTag':
        new_HIN_paras = ['MetaPath', None, None, None, 'TF_IDF']
    elif HIN_mode == 'EmbMaxText_MetaPathTag':  # 文本用EmbMax;tag用MetaPath；IsRec最基础
        new_HIN_paras = ['MetaPath', None, None, 'EmbMax', None]

    elif HIN_mode == 'IsRec_EmbMaxTag':
        new_HIN_paras = ['EmbMax', None, None, 'EmbMax', 'TF_IDF']  # IsRec_best tag用EmbMax,改写
    elif HIN_mode == 'IsRec_MetaPathTag':
        new_HIN_paras = ['MetaPath', None, None, 'EmbMax', 'TF_IDF']  # IsRec_best
    return new_HIN_paras


if __name__ == '__main__':
    # 模型
    new_old = 'old' # 'new','old','LR_PNCF','DIN_Rec'

    # 数据
    if_data_new = False
    data_mode = 'oldScene' # 'oldScene'  'newScene'
    need_slt_apis = False # False  True  新场景不用已选择服务时，设为False, 模型的handle_slt_apis_modes也设为False
    candidate_num = 'all' # 'all'
    if_fewSamples = False

    # 训练
    pairwise = False
    margin = 0.6
    train_mode = 'best_NDCG' # 'min_loss'用于快速调优，early stopping  'best_NDCG'每个epoch测试，很慢
    train_new = False  # 是否重新训练!!!训练多个组件的模型时最好设为true，利用之前的结果
    num_epochs = 10 # 10 !!! 对NI和MLP 改为10  CI可以为7
    CI_learning_rates = 0.0003 # 0.0001,0.0003,0.0005, pow(10,-2*i-2) for i in np.random.rand(5)
    NI_learning_rate = 0.0003# 0.0003
    topMLP_learning_rates=0.0001 # 0.0001,0.0003,0.0005,0.001
    l2_regs = 0 # 0.01,0.03不好 0.001  [0.0001,0.0005,0.001]
    embeddings_l2s = 0 # 1e-6,1e-5,1e-4,1e-3
    validation_split=0 # 验证集比例

    # CI
    content_fc_unit_nums = [200, 100, 50] # [200, 100, 50] > [100, 50] ,[256,64]不好
    emb_trains = True # 对比是否需要训练Embedding True,False
    inception_poolings = 'global_avg' # 'global_avg','global_max','max',,'none'
    if_inception_MLP = True # inception/textCNN后面是否跟MLP
    inception_MLP_dropout = True #  True
    inception_MLP_BN = False  # False
    text_extracter_modes =  ['inception']# 'inception' 'textCNN'  'HDP' 'LSTM'

    CI_handle_slt_apis_modes = False # 'attention' 'full_concate', 'average'  False # 新情景有用

    # NI
    train_mashup_best = False # !!! true是不正确做法，发现效果不好！
    NI_OL_mode = 'PasRec_2path' # 最新：'PasRec','PasRec_2path','IsRec','IsRec_best'   no_slts时用PasRec_2path

    # 'DeepCos' ,'HDPText_DeepTag','DeepText_MetaPathTag','HDPText_MetaPathTag'  'EmbMaxText_DeepTag'
    HIN_mode = 'DeepCos' # 主要针对文本  # 'DeepCos' 'EmbMaxText_MetaPathTag','TFIDFText_MetaPathTag' 'HDPText_MetaPathTag'
    new_HIN_paras = get_new_HIN_paras(HIN_mode)

    # implict结构！
    CF_self_1st_merge = True  # 测试ICI时，'pmf','nmf'应设为False；node2vec时，设为true!
    NI_handle_slt_apis_modes = [False] # # NI部分使用slt以及使用方法   [False,'attention','full_concate','average']
    cf_unit_nums = [100,50] # 隐式交互的MLP 两层够   # [200,100,50]   [100,50,25]
    topK_neighors=[5,10,20,30,40,50]  # 2 4 6 8  可参考IsRec_best的15
    mf_modes = ['node2vec'] # ,'pmf','nmf','node2vec','BiNE','BPR'

    predict_fc_unit_nums_list = [128,64,32]  # 200,100,50
    final_activation = 'softmax' # 'softmax'>'sigmoid'>pairwise  MSE?  测试PasRec等对照算法时用sigmoid

    # ****上面是默认设置，当设置不同时，初始化新的para_obj对象****
    para_obj = new_Para_(data_mode = 'newScene', need_slt_apis = False,
        topK = 20,num_epochs=10,CI_handle_slt_apis_mode= False,
                         NI_handle_slt_apis_mode = NI_handle_slt_apis_modes[0],NI_OL_mode= NI_OL_mode
                         )

    # para_obj = new_Para_(data_mode='oldScene', need_slt_apis=False, final_activation='sigmoid' ) # 针对PasRec2path
    new_Para.set_current_para(para_obj)  # 设置需要的参数对象
    meta_data.initilize()  # 初始化基础数据类，一定在new_Para.set_current_para()之后进行

    tst_kcv_paraObj(new_old, if_few=if_fewSamples)
