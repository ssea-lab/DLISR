# -*- coding:utf-8 -*-
import os
import sys

from recommend_models.DIN_Rec import DIN_Rec
from recommend_models.HINRec_new import HINRec_new
from recommend_models.MISR_models import top_MLP
from recommend_models.NI_Model_online_new import NI_Model_online

sys.path.append("..")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from recommend_models.HINRec_new import HINRec_new
# from run_deepCTR.new_run_MISR_deepFM import run_new_deepFM
from main.evalute import evalute_by_epoch, analyze_result
# from run_deepCTR.run_MISR_deepFM import run_deepFM
# from recommend_models.NI_Model_online_new import NI_Model_online

# from recommend_models.MISR_models import fine_Tune, top_MLP, PNCF_doubleTower,DINRec_model
# from recommend_models.PNCF import PNCF_model
# from recommend_models.set_rec import set_rec
from recommend_models.CI_Model import CI_Model
# from recommend_models.NI_Model import NI_Model
from main.new_para_setting import new_Para, new_Para_
from main.dataset import dataset, meta_data
from main.new_split_dataset import split_dataset_for_newScene_New_KCV, split_dataset_for_oldScene_KCV
from main.train_predict_methods import load_preTrained_model

# from recommend_models.set_rec import set_rec
from recommend_models.baseline import pop, TF_IDF, binary_keyword, Samanta, hdp_pop, MF

from recommend_models.recommend_Model import DHSR_model
# from recommend_models.text_only_model import gx_text_only_model, gx_text_only_MLP_model
# from recommend_models.text_tag_continue_model import gx_text_tag_continue_model, gx_text_tag_continue_only_MLP_model


def get_train_test_data(train_data, test_data,if_few = False,train_num= 64,test_num=32):
    # 是否使用小数据集先测试一下
    if if_few:
        train_data = (data[:train_num] for data in train_data)
        test_data = (data[:test_num] for data in test_data)
    return train_data, test_data


def baselines(a_dataset):
    # train_datas, test_datas = a_dataset.transfer_false_test_MF() # 新场景数据集微调，使得MF可以运行
    # modes = ['BPR'] # 'pmf',
    # for mode in modes:
    #     MF(train_datas, test_datas, mode)

    # pop()
    # TF_IDF(if_pop=False)
    # TF_IDF(if_pop=True)

    binary_keyword()
    # binary_keyword(True)
    # hdp_pop()

    # for k in [20, 30]:  # 10, 20, 30, 40, 50,100
    #     for if_pop in [2]: # 1,2
    #             for pop_mode in ['']:  # ,'sigmoid'
    #                 Samanta(k, if_pop=if_pop, pop_mode=pop_mode)

    # Samanta(30, if_pop=2, pop_mode='',text_mode='LDA',LDA_topic_num=20)
    # Samanta(30, if_pop=2, pop_mode='', text_mode='HDP')


def bl_set_rec():
    # set_rec
    set_rec('original_kmeans', embedding_ts=pow(10, -5), cluster_ts=pow(10, -4), cluster_num=10,
            embedding_mode='LDA').recommend([5])  # 'original_kmeans','manner_kmeans','manner'
    """
    cluster_tss = [2*pow(10, -5)] #[(1+i)*pow(10, -5) for i in range(5)] [pow(10, -3), pow(10, -4), pow(10, -5)]
    cluster_nums = [5]# range(5,10,2) [5,10] # , 30, 50
    for cluster_ts in cluster_tss:
        for cluster_num in cluster_nums:
            # set_rec('manner', embedding_ts=pow(10, -5), cluster_ts=pow(10, -4), cluster_num=cluster_num).recommend([5])  # 'original_kmeans','manner_kmeans','manner'
            set_rec('original_kmeans', embedding_ts=pow(10, -5), cluster_ts=cluster_ts,cluster_num=cluster_num,embedding_mode='HDP').recommend([5])
    """


# 完全冷启动
def bl_PNCF(a_dataset):

    mlp_modes = ['MLP']  # 'MLP',
    concate_modes = ['concate']  # 'multiply',,'concate'
    text_mode = 'HDP'
    for mlp_mode in mlp_modes:
        for concate_mode in concate_modes:
            pncf_rec_model = PNCF_model(mlp_mode=mlp_mode, concate_mode=concate_mode, text_mode=text_mode)
            pncf_model = pncf_rec_model.get_model()
            pncf_model = load_preTrained_model(pncf_rec_model, pncf_model, a_dataset.train_data,
                                               a_dataset.test_data, *new_Para.param.train_paras)


def text_only_oldScene(a_dataset):
    # 测试text_only model和only_MLP model
    text_recommend_model = gx_text_only_model()
    text_model = text_recommend_model.get_model()
    text_model = load_preTrained_model(text_recommend_model, text_model, a_dataset.train_data, a_dataset.test_data,
                                       *new_Para.param.train_paras)  # 'monitor loss&acc'
    print('train text_only_model, done!')

    # 基于text_only model的 text feature 继续训练模型
    text_only_MLP_recommend_model = gx_text_only_MLP_model('feature_cosine')  # 传入相似度计算方法
    text_only_MLP_recommend_model.prepare(text_recommend_model)
    text_only_MLP_model = text_only_MLP_recommend_model.get_model(text_model)
    load_preTrained_model(text_only_MLP_recommend_model, text_only_MLP_model, a_dataset.train_data, a_dataset.test_data,
                          *new_Para.param.train_paras)
    print('train gx_text_only_MLP_model, done!')

    """
    # 基于text_only model的 text feature + tag 继续训练模型
    text_tag_MLP_only_continue_recommend_model = gx_text_tag_continue_only_MLP_model(new_old,if_tag=False,if_HIN_sim=False,if_tag_sim=True)
    text_tag_MLP_only_continue_recommend_model.prepare(text_recommend_model)
    text_tag_MLP_only_continue_model = text_tag_MLP_only_continue_recommend_model.get_model()
    load_preTrained_model(text_tag_MLP_only_continue_recommend_model, text_tag_MLP_only_continue_model,
                          a_dataset.train_data, a_dataset.test_data, *new_Para.param.train_paras)
    text_tag_MLP_only_continue_recommend_model.save_sim_weight() # text和tag的相似度权值保存
    """


def bl_DHSR(a_dataset):
    dhsr_recommend_model = DHSR_model()
    dhsr_model = dhsr_recommend_model.get_model()

    # a_dataset.transfer() # 将重复sample删除？  'newScene'且need_slt_apis=False时

    train_data, test_data = get_train_test_data(a_dataset.train_data, a_dataset.test_data)
    dhsr_model = load_preTrained_model(dhsr_recommend_model, dhsr_model, train_data, test_data,*new_Para.param.train_paras)  # 'monitor loss&acc'
    dhsr_recommend_model.save_sth()
    evalute_by_epoch(dhsr_recommend_model, dhsr_model, dhsr_recommend_model.model_name,test_data)  # ,if_save_recommend_result=True,evaluate_by_slt_apiNum = True)


def bl_DHSR_new(a_dataset):
    train_datas, test_datas = a_dataset.transfer_false_test_DHSR(if_reduct_train=True)  # 是否约减训练集
    # 选择的服务数目不同，训练对应的模型，并评估效果
    for slt_num in range(1, new_Para.param.slt_item_num + 1):
        train_data, test_data = train_datas[slt_num - 1], test_datas[slt_num - 1]
        # old_new = 'new','new_sigmoid', 'new_reduct'效果最好
        dhsr_recommend_model = DHSR_model(old_new='new_reduct', slt_num=slt_num)
        dhsr_model = dhsr_recommend_model.get_model()
        dhsr_model = load_preTrained_model(dhsr_recommend_model, dhsr_model, train_data, test_data,*new_Para.param.train_paras)  # 'monitor loss&acc'
        evalute_by_epoch(dhsr_recommend_model, dhsr_model, dhsr_recommend_model.model_name, test_data, evaluate_by_slt_apiNum=True)
        dhsr_recommend_model.save_sth()
        print('DHSR, slt_num:{}, train_predict,done!'.format(slt_num))


def bl_PasRec(a_dataset):
    model_name = 'PasRec_2path'  # 'PasRec_2path'
    epoch_num = 20  # 之前是40  40比20差点
    neighbor_size = 15
    topTopicNum = 3

    train_data, test_data = get_train_test_data(a_dataset.train_data, a_dataset.test_data)
    HINRec_model = HINRec_new(model_name=model_name, epoch_num=epoch_num, neighbor_size=neighbor_size,topTopicNum=topTopicNum)

    # 使用LDA处理PasRec的相似度   50 100 150
    # HINRec_model = HINRec_new(model_name=model_name, semantic_mode='LDA', LDA_topic_num=50, epoch_num=epoch_num,
    #                           neighbor_size=neighbor_size,
    #                           topTopicNum=topTopicNum)
    if os.path.exists(HINRec_model.weight_path):
        print('have trained,return!')
    else:
        # 这里是每隔20epoch测试一下，所以train中输入test_data
        HINRec_model.train(test_data)
        HINRec_model.save_model()

        evalute_by_epoch(HINRec_model, HINRec_model, HINRec_model.model_name,test_data,evaluate_by_slt_apiNum = True)  # ,if_save_recommend_result=True)


def bl_IsRec(a_dataset):
    model_name = 'IsRec'  # ''
    epoch_nums = [20]  # 15,100,1000
    neighbor_size = 15
    topTopicNums = [3]  # [3,4,5,6]

    train_data, test_data = get_train_test_data(a_dataset.train_data, a_dataset.test_data)

    for epoch_num in epoch_nums:
        for topTopicNum in topTopicNums:
            HINRec_model = HINRec_new(model_name=model_name, epoch_num=epoch_num, neighbor_size=neighbor_size,topTopicNum=topTopicNum)

            if os.path.exists(HINRec_model.weight_path):
                print('have trained,return!')
            else:
                HINRec_model.train(test_data)
                # HINRec_model.test_model(test_data)
                HINRec_model.save_model()

                evalute_by_epoch(HINRec_model, HINRec_model, HINRec_model.model_name, test_data,evaluate_by_slt_apiNum=True)  # ,if_save_recommend_result=True)


def bl_IsRec_best(a_dataset):
    model_name = 'IsRec_best'  # 'IsRec'  'IsRec_best_modified'
    epoch_num = 20
    neighbor_size = 15
    topTopicNum = 3
    cluster_mode = 'LDA'
    cluster_mode_topic_nums = [50]  # 10,25,75,,100,125,150
    train_data, test_data = get_train_test_data(a_dataset.train_data, a_dataset.test_data)
    for cluster_mode_topic_num in cluster_mode_topic_nums:
        HINRec_model = HINRec_new(model_name=model_name, semantic_mode='TF_IDF', epoch_num=epoch_num,
                                  neighbor_size=neighbor_size, topTopicNum=topTopicNum, cluster_mode=cluster_mode,
                                  cluster_mode_topic_num=cluster_mode_topic_num)

        if os.path.exists(HINRec_model.weight_path):
            print('have trained,return!')
        else:
            HINRec_model.train(test_data)
            HINRec_model.save_model()

            evalute_by_epoch(HINRec_model, HINRec_model, HINRec_model.model_name,test_data, evaluate_by_slt_apiNum=True)  # )
            # analyze_result(HINRec_model, new_Para.param.topKs)


def text_tag():
    # # 获得text_tag_model
    text_tag_recommend_model = gx_text_tag_continue_model(new_old)
    text_tag_recommend_model.prepare()
    text_tag_model = text_tag_recommend_model.get_model()

    # 训练测试
    text_tag_model = load_preTrained_model(text_tag_recommend_model, text_tag_model, train_data, test_data,
                                           *new_Para.param.train_paras)  # 'monitor loss&acc'

    text_tag_recommend_model.show_slt_apis_tag_features(a_dataset.train_data)
    print('show_slt_apis_tag_features, done!')

    if if_whole:
        # 整个模型
        text_tag_MLP_only_continue_recommend_model = gx_text_tag_continue_only_MLP_model(new_old)
        # 为搭建模型做准备
        text_tag_MLP_only_continue_recommend_model.prepare(text_tag_recommend_model)
        text_tag_MLP_only_continue_model = text_tag_MLP_only_continue_recommend_model.get_model(text_tag_model)

        load_preTrained_model(text_tag_MLP_only_continue_recommend_model, text_tag_MLP_only_continue_model, train_data,
                              test_data, *new_Para.param.train_paras)
        """
        # 新场景下针对每种个数的进行测试
        testsets = dataset.crt_ds.split_test_ins_by_sltNum()
        for index in range(len(testsets)):
            slt_api_num = index + 1
            test_data = testsets[index]
            evalute_by_epoch(text_tag_MLP_only_continue_recommend_model, text_tag_MLP_only_continue_model,
                             text_tag_MLP_only_continue_recommend_model.get_name(), test_data,
                             by_slt_num=slt_api_num)
        """


def CI_NI_fineTuning():
    CI_recommend_model = CI_Model('new')  # new_old
    # CI_recommend_model.prepare()
    # CI_model_obj = CI_recommend_model.get_model()
    # CI_model_obj = load_preTrained_model(CI_recommend_model, CI_model_obj,train_data,test_data,*new_Para.param.train_paras)  # 'monitor loss&acc'
    # CI_recommend_model.show_slt_apis_tag_features(a_dataset.train_data) # 存储中间feature结果

    # implict
    implict_NI_recommend_model = NI_Model('new', if_implict=True, if_explict=False, if_correlation=False)  # new_old
    implict_NI_recommend_model.prepare(CI_recommend_model)  # NI的模型搭建需要CI模型生成所有mashup/api的feature
    implict_NI_model_obj = implict_NI_recommend_model.get_model()
    implict_NI_model_obj = load_preTrained_model(implict_NI_recommend_model, implict_NI_model_obj, train_data,
                                                 test_data, *new_Para.param.train_paras)

    # # explicit
    # explicit_NI_recommend_model = NI_Model(new_old,if_implict=False,if_explict=True,if_correlation=False)
    # explicit_NI_recommend_model.prepare(CI_recommend_model) # NI的模型搭建需要CI模型生成所有mashup/api的feature
    # explict_NI_model_obj = explicit_NI_recommend_model.get_model()
    # explict_NI_model_obj = load_preTrained_model(explicit_NI_recommend_model, explict_NI_model_obj,train_data, test_data, *new_Para.param.train_paras)

    # # # ENI+INI一起训练！！！训练这个时，不需要设置explicit/implicit
    # NI_recommend_model = NI_Model(new_old,if_implict=True,if_explict=True,if_correlation=False)
    # NI_recommend_model.prepare(CI_recommend_model) # NI的模型搭建需要CI模型生成所有mashup/api的feature
    # NI_model_obj = NI_recommend_model.get_model()
    # NI_model_obj = load_preTrained_model(NI_recommend_model, NI_model_obj,train_data, test_data, *new_Para.param.train_paras)

    # # # cor
    # cor_NI_recommend_model = NI_Model(new_old,if_implict=False,if_explict=False,if_correlation=True)
    # cor_NI_recommend_model.prepare(CI_recommend_model) # NI的模型搭建需要CI模型生成所有mashup/api的feature
    # cor_NI_model_obj = cor_NI_recommend_model.get_model()
    # cor_NI_model_obj = load_preTrained_model(cor_NI_recommend_model, cor_NI_model_obj,train_data, test_data, *new_Para.param.train_paras)

    # # CI+ implict
    # top_MLP_recommend_model = top_MLP(CI_recommend_model,CI_model_obj,NI_recommend_model=implict_NI_recommend_model,NI_model=implict_NI_model_obj)

    # # CI+ explicit
    # top_MLP_recommend_model = top_MLP(CI_recommend_model, CI_model_obj,NI_recommend_model2=explicit_NI_recommend_model,NI_model2=explict_NI_model_obj)

    #
    # # 大组合
    # top_MLP_recommend_model = top_MLP(CI_recommend_model=None, CI_model=None,CI_model_dir=CI_recommend_model.model_dir,
    #                                   NI_recommend_model=implict_NI_recommend_model,
    #                                   NI_model=implict_NI_model_obj,
    #                                   NI_recommend_model2 = explicit_NI_recommend_model,
    #                                   NI_model2=explict_NI_model_obj,new_old=new_old
    #                                   ) # cor_NI_recommend_model,cor_NI_model_obj
    #
    # top_MLP_model = top_MLP_recommend_model.get_model()
    # top_MLP_model = load_preTrained_model(top_MLP_recommend_model, top_MLP_model, train_data, test_data,*new_Para.param.train_paras)
    # top_MLP_recommend_model.save_sth() # 存储训练测试过程中使用的所有实例的中间结果

    # fine_Tune_mode = '3MLP_att'
    # fineTune_recommend_model = fine_Tune(CI_recommend_model,NI_recommend_model,top_MLP_recommend_model,CI_model_obj,NI_model_obj,top_MLP_model,mode=fine_Tune_mode)
    # fineTune_model = fineTune_recommend_model.get_model()
    # fineTune_model = load_preTrained_model(fineTune_recommend_model, fineTune_model, train_data, test_data,*new_Para.param.train_paras)


def NI_online():  # 可以用于CI,NI,topMLP,ft等
    # HINRec_model = HINRec(model_name='IsRec_best',semantic_mode='TF_IDF', epoch_num=40, neighbor_size=15,topTopicNum=3,cluster_mode='LDA',cluster_mode_topic_num=50)
    HINRec_model = HINRec(model_name='PasRec', epoch_num=40, neighbor_size=15, topTopicNum=3)

    CI_recommend_model = CI_Model(new_old)  # 'old'
    # CI_recommend_model.prepare()
    # CI_model_obj = CI_recommend_model.get_model()
    # CI_model_obj = load_preTrained_model(CI_recommend_model, CI_model_obj,train_data,test_data,*new_Para.param.train_paras)  # ,true_candidates_dict=HINRec_model.get_true_candi_apis() 'monitor loss&acc'
    # evalute_by_epoch(CI_recommend_model, CI_model_obj, CI_recommend_model.model_name, test_data,
    #                  if_save_recommend_result=True, evaluate_by_slt_apiNum=True)
    # analyze_result(CI_recommend_model, new_Para.param.topKs)

    # CI_recommend_model.show_slt_apis_tag_features(a_dataset.train_data) # 检查中间feature结果
    # # CI_recommend_model.get_slt_apis_mid_features(train_data,test_data) # 存储所有样本的attention的中间结果,为deepFm准备
    # # CI_recommend_model.save_for_deepFM()
    #

    # 调优NI的score
    # for pruned_neighbor_baseScore in [0,0.2,0.3]: #
    #     NI_OL_recommend_model = NI_Model_online(new_old, if_implict=True, if_explict=False,
    #                                             if_correlation=False,
    #                                             pruned_neighbor_baseScore=pruned_neighbor_baseScore)
    #     sim_model = HINRec_model if new_Para.param.NI_OL_mode == 'IsRec_best_Sim' else None  # CI_recommend_model
    #     NI_OL_recommend_model.prepare(sim_model, train_data, test_data)
    #     NI_OL_model_obj = NI_OL_recommend_model.get_model()
    #     NI_OL_model_obj = load_preTrained_model(NI_OL_recommend_model, NI_OL_model_obj, train_data, test_data,
    #                                             *new_Para.param.train_paras,
    #                                             true_candidates_dict=HINRec_model.get_true_candi_apis())

    NI_OL_recommend_model = NI_Model_online('new', if_implict=True, if_explict=False,
                                            if_correlation=False)  # 'new' ,pruned_neighbor_baseScore = 0
    # 构建即可，读取之前训练好的相似度数据
    # HINRec_model = HINRec(model_name='IsRec_best',semantic_mode='TF_IDF', epoch_num=40, neighbor_size=15,topTopicNum=3,cluster_mode='LDA',cluster_mode_topic_num=50)
    # 'IsRec_best_Sim'
    sim_model = CI_recommend_model if new_Para.param.NI_OL_mode == 'tagSim' else HINRec_model  #
    NI_OL_recommend_model.prepare(sim_model, train_data, test_data)
    NI_OL_model_obj = NI_OL_recommend_model.get_model()
    NI_OL_model_obj = load_preTrained_model(NI_OL_recommend_model, NI_OL_model_obj, train_data, test_data,
                                            *new_Para.param.train_paras)  # ,true_candidates_dict=HINRec_model.get_true_candi_apis()
    # # dataset.crt_ds.UV_obj.save_onlineNode2vec()
    # # evalute_by_epoch(NI_OL_recommend_model, NI_OL_model_obj, 'NI_IsRec_true_candidates', test_data,record_time=False, true_candidates_dict=HINRec_model.get_true_candi_apis())
    # # NI_OL_recommend_model.get_slt_apis_mid_features(train_data, test_data) # 存储所有样本的attention的中间结果,为deepFm准备
    # # NI_OL_recommend_model.save_for_deepFM()

    # evalute_by_epoch(NI_OL_recommend_model, NI_OL_model_obj, NI_OL_recommend_model.model_name, test_data,if_save_recommend_result=True, evaluate_by_slt_apiNum=True)
    # analyze_result(NI_OL_recommend_model, new_Para.param.topKs)
    #
    # # # CI+ implict
    # top_MLP_recommend_model = top_MLP(CI_recommend_model,CI_model_obj,NI_recommend_model=NI_OL_recommend_model,NI_model=NI_OL_model_obj)
    # top_MLP_model = top_MLP_recommend_model.get_model()
    # top_MLP_model = load_preTrained_model(top_MLP_recommend_model, top_MLP_model, train_data, test_data,*new_Para.param.train_paras)
    # top_MLP_recommend_model.save_sth() # 存储训练测试过程中使用的所有实例的中间结果
    #
    # evalute_by_epoch(top_MLP_recommend_model, top_MLP_model, top_MLP_recommend_model.model_name, test_data,
    #                  if_save_recommend_result=True, evaluate_by_slt_apiNum=True)
    # analyze_result(top_MLP_recommend_model, new_Para.param.topKs)
    #
    # # # 测试使用Isrec_best的评价技术
    # evalute_by_epoch(top_MLP_recommend_model, top_MLP_model, 'MLP_IsRec_true_candidates', test_data, record_time=False,true_candidates_dict=HINRec_model.get_true_candi_apis())

    # #
    # fine_Tune_mode = '3MLP' # 'whole'  '3MLP'
    # fineTune_recommend_model = fine_Tune(CI_recommend_model, CI_model_obj,NI_OL_recommend_model,NI_OL_model_obj, top_MLP_recommend_model, top_MLP_model,
    #                                      model_mode='ft',ft_mode=fine_Tune_mode)
    # fineTune_model = fineTune_recommend_model.get_model()
    # fineTune_recommend_model.pre_fine_tune()
    # fineTune_model = load_preTrained_model(fineTune_recommend_model, fineTune_model, train_data, test_data,*new_Para.param.train_paras)


def deepFM():
    CI_recommend_model = CI_Model(new_old)  # 'old'
    NI_OL_recommend_model = NI_Model_online(new_old, if_implict=True, if_explict=False, if_correlation=False)
    # 上面仅初始化，否则加载模型会占用太多现存
    run_deepFM(CI_recommend_model, NI_OL_recommend_model, train_data, test_data, epoch_num=10)


def get_preTrain_CINI_model():
    HINRec_model = HINRec(model_name='PasRec', epoch_num=40, neighbor_size=15, topTopicNum=3)
    CI_recommend_model = CI_Model(new_old)  # 'old'
    CI_recommend_model.prepare()
    CI_model_obj = CI_recommend_model.get_model()
    CI_model_obj = load_preTrained_model(CI_recommend_model, CI_model_obj, train_data, test_data,
                                         *new_Para.param.train_paras)  # ,true_candidates_dict=HINRec_model.get_true_candi_apis() 'monitor loss&acc'

    NI_OL_recommend_model = NI_Model_online(new_old, if_implict=True, if_explict=False,
                                            if_correlation=False)  # 'new' ,pruned_neighbor_baseScore = 0
    # 构建即可，读取之前训练好的相似度数据
    # HINRec_model = HINRec(model_name='IsRec_best',semantic_mode='TF_IDF', epoch_num=40, neighbor_size=15,topTopicNum=3,cluster_mode='LDA',cluster_mode_topic_num=50)
    # 'IsRec_best_Sim'
    sim_model = CI_recommend_model if new_Para.param.NI_OL_mode == 'tagSim' else HINRec_model  #
    NI_OL_recommend_model.prepare(sim_model, train_data, test_data)

    return CI_recommend_model, NI_OL_recommend_model


def newDeepFM():
    CI_recommend_model = CI_Model(new_old)  # 'old'
    CI_recommend_model.prepare()

    HINRec_model = HINRec(model_name='PasRec', epoch_num=40, neighbor_size=15, topTopicNum=3)
    sim_model = CI_recommend_model if new_Para.param.NI_OL_mode == 'tagSim' else HINRec_model  #
    NI_OL_recommend_model = NI_Model_online(new_old, if_implict=True, if_explict=False,
                                            if_correlation=False)  # 'new' ,pruned_neighbor_baseScore = 0
    NI_OL_recommend_model.prepare(sim_model, train_data, test_data)
    mashup_NI_features = NI_OL_recommend_model.mid_sltAids_2NI_feas
    api_NI_features = NI_OL_recommend_model.i_factors_matrix
    NI_feas = mashup_NI_features, api_NI_features

    if not os.path.exists(CI_recommend_model.ma_text_tag_feas_path):
        # 如果特征的存储文件不存在，再加载模型，退出重新运行
        CI_model_obj = CI_recommend_model.get_model()
        CI_model_obj = load_preTrained_model(CI_recommend_model, CI_model_obj, train_data, test_data,
                                             *new_Para.param.train_paras)  # ,true_candidates_dict=HINRec_model.get_true_candi_apis() 'monitor loss&acc'
        CI_feas = CI_recommend_model.get_mashup_api_features(CI_recommend_model.all_mashup_num,
                                                             CI_recommend_model.all_api_num + 1)  # 最后一个是填充虚拟api的特征
        print('re-run the program!')

    else:
        CI_feas = CI_recommend_model.get_mashup_api_features(CI_recommend_model.all_mashup_num,
                                                             CI_recommend_model.all_api_num + 1)

        run_new_deepFM(CI_feas, NI_feas, train_data, test_data, CI_recommend_model.all_api_num, epoch_num=10)


def co_trainCINI():  # 参数完全随机化，联合训练CI和NI
    CI_recommend_model = CI_Model(new_old)
    CI_recommend_model.prepare()
    CI_model_obj = CI_recommend_model.get_model()

    NI_OL_recommend_model = NI_Model_online(new_old, if_implict=True, if_explict=False, if_correlation=False)
    # CI_recommend_model = CI_recommend_model if 'Sim' in new_Para.param.NI_OL_mode else None 只训练NI时，OL_GE不需要CI
    NI_OL_recommend_model.prepare(CI_recommend_model)
    NI_OL_model_obj = NI_OL_recommend_model.get_model()

    co_trainCINI_recommend_model = fine_Tune(CI_recommend_model, CI_model_obj, NI_OL_recommend_model, NI_OL_model_obj,
                                             model_mode='co_train', lr=0.001)  # 0.0003
    co_trainCINI_model = co_trainCINI_recommend_model.get_model()
    co_trainCINI_model = load_preTrained_model(co_trainCINI_recommend_model, co_trainCINI_model, train_data, test_data,
                                               *new_Para.param.train_paras)


def test_PNCF_doubleTower_OR_DIN():
    CI_recommend_model, NI_OL_recommend_model = get_preTrain_CINI_model()

    # PNCF_recommend_model = PNCF_doubleTower(CI_recommend_model= CI_recommend_model,CI_model=CI_model_obj,NI_recommend_model=NI_OL_recommend_model,NI_model=NI_OL_model_obj)
    # PNCF_model_obj = PNCF_recommend_model.get_model()
    # PNCF_model_obj = load_preTrained_model(PNCF_recommend_model, PNCF_model_obj, train_data, test_data,*new_Para.param.train_paras)

    # evalute_by_epoch(PNCF_recommend_model, PNCF_model_obj, PNCF_recommend_model.model_name, test_data, if_save_recommend_result=True, evaluate_by_slt_apiNum=True)
    # analyze_result(PNCF_recommend_model, new_Para.param.topKs)

    # model_name = 'MLP_embedding'!!! 'DINRec'
    DINRec_recommend_model = DINRec_model(CI_recommend_model=CI_recommend_model,
                                          NI_recommend_model=NI_OL_recommend_model)  # !!! ,model_name='MLP_embedding'
    DINRec_model_obj = DINRec_recommend_model.get_model()
    DINRec_model_obj = load_preTrained_model(DINRec_recommend_model, DINRec_model_obj, train_data, test_data,
                                             *new_Para.param.train_paras)

    # evalute_by_epoch(DINRec_recommend_model, DINRec_model_obj, DINRec_recommend_model.model_name, test_data, if_save_recommend_result=True, evaluate_by_slt_apiNum=True)
    # analyze_result(DINRec_recommend_model, new_Para.param.topKs)

def DINRec(a_dataset,new_old = 'new'):
    train_data, test_data = a_dataset.train_data, a_dataset.test_data
    CI_recommend_model = CI_Model(new_old)  # 'old'
    CI_recommend_model.prepare()
    CI_model_obj = CI_recommend_model.get_model()
    CI_model_obj = load_preTrained_model(CI_recommend_model, CI_model_obj,train_data,test_data,*new_Para.param.train_paras)  # ,true_candidates_dict=HINRec_model.get_true_candi_apis() 'monitor loss&acc'

    DINRec_model = DIN_Rec(CI_recommend_model,new_Para.param.predict_fc_unit_nums)
    DINRec_model.prepare()
    DINRec_model_obj = DINRec_model.get_model()
    DINRec_model_obj = load_preTrained_model(DINRec_model, DINRec_model_obj,train_data,test_data,*new_Para.param.train_paras)  # ,true_candidates_dict=HINRec_model.get_true_candi_apis() 'monitor loss&acc'
    evalute_by_epoch(DINRec_model, DINRec_model_obj, DINRec_model.simple_name, test_data,
                        if_save_recommend_result=True, evaluate_by_slt_apiNum=True)

def test_simModes(a_dataset,new_old = 'new',if_few = False):
    if if_few:
        train_data, test_data = a_dataset.get_few_samples(128)
        print(type(train_data))
        print(type(test_data))
    else:
        train_data, test_data = a_dataset.train_data,a_dataset.test_data

    HINRec_model = HINRec_new(model_name=new_Para.param.NI_OL_mode, epoch_num=20, neighbor_size=15, topTopicNum=3)
    # 'IsRec_best' 这个是预训练的相似度模型
    CI_recommend_model = CI_Model(new_old)  # 'old'
    CI_recommend_model.prepare()
    CI_model_obj = CI_recommend_model.get_model()
    CI_model_obj = load_preTrained_model(CI_recommend_model, CI_model_obj,train_data,test_data,*new_Para.param.train_paras)  # ,true_candidates_dict=HINRec_model.get_true_candi_apis() 'monitor loss&acc'
    # evalute_by_epoch(CI_recommend_model, CI_model_obj, CI_recommend_model.simple_name, test_data,
    #                     if_save_recommend_result=True, evaluate_by_slt_apiNum=True)

    # 为no_slt CI设计
    # CI_model_obj = load_preTrained_model(CI_recommend_model, CI_model_obj, train_data, a_dataset.test_data_no_reduct,
    #                                         *new_Para.param.train_paras)  #
    # evalute_by_epoch(CI_recommend_model, CI_model_obj, CI_recommend_model.simple_name, a_dataset.test_data_no_reduct,
    #                  if_save_recommend_result=True, evaluate_by_slt_apiNum=True)

    sim_model = CI_recommend_model if new_Para.param.NI_OL_mode == 'tagSim' else HINRec_model  #
    NI_OL_recommend_model = NI_Model_online(new_old, if_implict=True, if_explict=False, if_correlation=False,
                                            eachPath_topK=True)  # 'new' ,pruned_neighbor_baseScore = 0
    # 构建即可，读取之前训练好的相似度数据
    NI_OL_recommend_model.prepare_sims(sim_model, train_data, test_data)
    NI_OL_model_obj = NI_OL_recommend_model.get_model()
    NI_OL_model_obj = load_preTrained_model(NI_OL_recommend_model, NI_OL_model_obj, train_data, test_data,
                                            *new_Para.param.train_paras)  #
    # # # explicit
    # explicit_NI_recommend_model = NI_Model_online(new_old,if_implict=False,if_explict=True,if_correlation=False)
    # explicit_NI_recommend_model.prepare(sim_model,train_data, test_data) # NI的模型搭建需要CI模型生成所有mashup/api的feature
    # explict_NI_model_obj = explicit_NI_recommend_model.get_model()
    # explict_NI_model_obj = load_preTrained_model(explicit_NI_recommend_model, explict_NI_model_obj,train_data, test_data, *new_Para.param.train_paras)

    # evalute_by_epoch(NI_OL_recommend_model, NI_OL_model_obj, NI_OL_recommend_model.simple_name, a_dataset.test_data,
    #                  if_save_recommend_result=True, evaluate_by_slt_apiNum=True)

    # 专门为no_slt NI设计
    # NI_OL_model_obj = load_preTrained_model(NI_OL_recommend_model, NI_OL_model_obj, train_data, a_dataset.test_data_no_reduct,
    #                                         *new_Para.param.train_paras)  #
    # evalute_by_epoch(NI_OL_recommend_model, NI_OL_model_obj, NI_OL_recommend_model.simple_name, a_dataset.test_data_no_reduct,
    #                  if_save_recommend_result=True, evaluate_by_slt_apiNum=True)
    # #
    # # # # # CI+ implict
    top_MLP_recommend_model = top_MLP(CI_recommend_model,CI_model_obj,NI_recommend_model=NI_OL_recommend_model,NI_model=NI_OL_model_obj)
    top_MLP_model = top_MLP_recommend_model.get_model()
    top_MLP_model = load_preTrained_model(top_MLP_recommend_model, top_MLP_model, train_data, test_data,*new_Para.param.train_paras)
    top_MLP_recommend_model.save_sth() # 存储训练测试过程中使用的所有实例的中间结果
    # evalute_by_epoch(top_MLP_recommend_model, top_MLP_model, top_MLP_recommend_model.simple_name, test_data,
    #                     if_save_recommend_result=True, evaluate_by_slt_apiNum=True)
    # analyze_result(top_MLP_recommend_model, new_Para.param.topKs)
    #
    # top_MLP_recommend_model = top_MLP(CI_recommend_model=None, CI_model=None,CI_model_dir=CI_recommend_model.model_dir,
    #                                   NI_recommend_model=NI_OL_recommend_model,
    #                                   NI_model=NI_OL_model_obj,
    #                                   NI_recommend_model2 = explicit_NI_recommend_model,
    #                                   NI_model2=explict_NI_model_obj,new_old=new_old
    #                                   ) # cor_NI_recommend_model,cor_NI_model_obj

    # top_MLP_model = top_MLP_recommend_model.get_model()
    # top_MLP_model = load_preTrained_model(top_MLP_recommend_model, top_MLP_model, train_data, test_data,*new_Para.param.train_paras)
    # top_MLP_recommend_model.save_sth() # 存储训练测试过程中使用的所有实例的中间结果

def test():
    HINRec_model = HINRec(model_name='IsRec_best', semantic_mode='TF_IDF', epoch_num=40, neighbor_size=15,
                          topTopicNum=3, cluster_mode='LDA', cluster_mode_topic_num=50)
    print(HINRec_model.get_true_candi_apis())
