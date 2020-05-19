import os
import sys
sys.path.append("..")


class new_Para_(object):
    def __init__(self, embedding_train=True, num_epochs=10, CI_learning_rate=0.0003,NI_learning_rate=0.0003,topMLP_learning_rate=0.0001, l2_reg=0, mf_mode='node2vec',
                     data_mode='newScene', need_slt_apis=True, if_implict=True, if_explict=3, if_correlation=False,
                     train_new=False,batch_size = 32,if_inception_MLP = True, content_fc_unit_nums=[200,100,50],
                     inception_pooling='global_avg',candidate_num='all',text_extracter_mode ='inception',
                     topK = 50,CF_self_1st_merge=True, NI_handle_slt_apis_mode='attention',CI_handle_slt_apis_mode='attention',
                     cf_unit_nums = [100,50],predict_fc_unit_nums=[128,64,32],HIN_mode='DeepCos',new_HIN_paras = [None, None, 'Deep', None, 'Deep'],
                     inception_MLP_dropout = True,inception_MLP_BN = False,train_mode='best_NDCG',pairwise=False,margin=0.6,embeddings_l2=0,final_activation='softmax',
                     validation_split=0,NI_OL_mode='PasRec',train_mashup_best = False,if_data_new= False):

        # 文本
        self.remove_punctuation = False # 文本是否去除标点
        self.embedding_name = 'glove'
        self.embedding_dim = 50
        self.MAX_SEQUENCE_LENGTH = 150
        self.MAX_NUM_WORDS = 30000
        self.keras_filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
        self.stop_words = set(['!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'])  # 去标点符号？？

        # 模型搭建
        # func部分
        self.text_extracter_mode = text_extracter_mode # 'inception','LSTM','textCNN'
        self.inception_channels =   [10, 10, 10, 20, 10]#   [64, 96, 64, 96, 64]
        self.textCNN_channels = [20,20,20] # [128,128,128]
        self.inception_pooling = inception_pooling  # 'global_max' 'max' 'global_avg','none'
        self.LSTM_dim = 25 # 128 64
        self.if_inception_MLP = if_inception_MLP
        self.inception_fc_unit_nums = [100, 50]  # inception后接的FC
        self.inception_MLP_dropout = inception_MLP_dropout
        self.inception_MLP_BN = inception_MLP_BN #

        self.Category_type = 'all' # 'first'
        self.merge_manner = 'direct_merge' # 'final_merge',
        self.mf_mode = mf_mode # MF部分使用哪种方法的向量

        # merge_manner='final_merge'时有用
        self.text_fc_unit_nums = [100, 50] # 文本向量和类别向量分别MLP后merge再MLP的设置：[100,50]
        self.tag_fc_unit_nums = [100, 50]
        self.content_fc_unit_nums = content_fc_unit_nums # 处理文本MLP的维度 # [200,100,50,25]  [100, 50]   [256,64,16,8] [1024,256,64]

        self.topK = topK # 最近邻数目

        self.CI_handle_slt_apis_mode = CI_handle_slt_apis_mode
        # 使用哪种拼接方式得到的CI
        if self.CI_handle_slt_apis_mode == 'attention':
            self.simple_CI_mode = 'att'
        elif self.CI_handle_slt_apis_mode == 'average':
            self.simple_CI_mode = 'ave'
        elif self.CI_handle_slt_apis_mode == 'full_concate':
            self.simple_CI_mode = 'ful'
        else: # 不选时设为False
            self.simple_CI_mode = 'no_slts'

        self.NI_OL_mode = NI_OL_mode
        # 隐式交互部分
        self.if_implict = if_implict
        self.NI_handle_slt_apis_mode = NI_handle_slt_apis_mode
        self.num_feat = 25 # 隐式交互部分的隐向量维度
        self.cf_unit_nums = cf_unit_nums  # 隐式CF交互中，UV先整合再跟pair整合
        self.CF_self_1st_merge = CF_self_1st_merge # 隐式表示是否先用MLP处理，还是元素乘

        # 显式交互
        self.if_explict= if_explict # 不同数字代表几种不同模式
        self.deep_co_fc_unit_nums = [1024, 256, 64, 16] # 显式交互部分的MLP
        self.shadow_co_fc_unit_nums = [64,16] # 128,  0-1上是3层，2-4是2层！！！

        # 可组合性
        self.if_correlation = if_correlation
        self.cor_fc_unit_nums = [128,64,16]

        self.predict_fc_unit_nums = predict_fc_unit_nums  # 整合各部分后最后的MLP 32,16,8   200,100,50,25 整合MF后预测时层数太深？
        self.final_activation = final_activation

        self.sim_feature_size = 8 # DHSR model
        self.DHSR_layers1 = [32, 16, 8]
        self.DHSR_layers2 = [32, 16, 8]
        self.mf_embedding_dim = 50  # 8 用于DHSR等模型!!!
        self.mf_fc_unit_nums = [120, 50]  # mf部分最终维度 #  32,16,8   120,50

        self.NCF_layers = [64, 32, 16, 8]
        self.NCF_reg_layers = [0.01, 0.01, 0.01, 0.01]
        self.NCF_reg_mf = 0.01

        # 模型训练
        self.embedding_train = embedding_train
        self.embeddings_regularizer = embeddings_l2 # 1e-6!!!
        self.num_epochs = num_epochs
        self.CI_learning_rate = CI_learning_rate
        self.NI_learning_rate = NI_learning_rate
        self.topMLP_learning_rate = topMLP_learning_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.sss_batch_size = 16
        self.small_batch_size = 32
        self.big_batch_size = 64
        self.huge_batch_size = 128
        self.train_mode = train_mode # 'best_NDCG' # 'min_loss'
        self.train_new = train_new
        self.validation_split=validation_split
        # 是否使用pairwise型训练！！！
        self.pairwise=pairwise
        self.margin = margin #

        # 模型评估
        self.topKs = [k for k in range(5, 55, 5)]  # 测试NDCG用

        # 重要路径
        self.root_path = os.path.abspath('..')  # 表示当前所处的文件夹的绝对路径,针对新代码而言！
        features_result_path = os.path.join(self.root_path, 'feature')
        history_result_path = os.path.join(self.root_path, 'history')
        self.if_data_new = if_data_new
        data_path = 'new_data' if if_data_new else 'data'
        self.data_dir = os.path.join(self.root_path, data_path) # !!!新数据旧数据

        # 'evaluate_result.csv' 'evaluate_result_BiNE.csv' 'evaluate_result_OurIs.csv'
        # 'evaluate_result_textCNN.csv' 'evaluate_result_LSTM.csv' 'textExtracter' 'implict' 'EMLP' '_sim1'  ‘ablation’ 'newScene_context' 'EarlyStop' oldConverge

        # evaluate_acc_newScene_context_ + OLGE tagSim topMLP ft_3MLP ft_whole co_train _lr0003  full noSlts  evaluate_result_newScene_PasRec average channel
        # 'evaluate_result_newScene_NEW_dataset_com6_neg14_CI.csv'  PasRec2Path  IsRecBest oldScene_CI  newScene_CI_oldModel
        #  baselines
        self.evaluate_path = os.path.join(self.data_dir, 'evaluate_new_CI.csv') # _cotrain  3mode pasRec2path
        self.loss_path = os.path.join(self.data_dir, 'loss_acc_new_CI.csv') # OLGE deepFM CIIsRecMLP TrueChannelAtt2-1 PartTarget_MLP3 IsRec_best
        self.time_path = os.path.join(self.data_dir, 'time_new_CI.csv') # OLGE
        self.glove_embedding_path = r'../data/pre_trained_embeddings'
        self.google_embedding_path = r''

        self.data_mode = data_mode # 确定新场景下的数据集  包含不同组合下的多种正例
        self.need_slt_apis = need_slt_apis  # 确定在使用数据集时是否使用slt api的数据   新数据也可以用在旧模型中，只需要设为false即可

        # 旧场景划分下的参数...待改
        self.num_negatives = 12  # 6 正常是6，为加速调试，设为2
        self.split_mannner = 'cold_start'  # 'cold_start' 冷启动问题研究 'left_one_spilt'一般问题最佳,'left_one_spilt','mashup'
        self.train_ratio = 0.7
        self.candidates_manner = 'all'  # 'num' 'ratio'  'all'
        self.candidates_num =  candidate_num # 100  # 100
        self.candidates_ratio = 99
        self.s = ''  # 名称中的取值字段
        if self.candidates_manner == 'ratio':
            self.s = self.candidates_ratio
        elif self.candidates_manner == 'num':
            self.s = self.candidates_num
        self.data_name = '{}_{}_{}_{}_{}'.format(self.split_mannner, self.train_ratio, self.candidates_manner, self.s, self.num_negatives)

        # 新场景数据集使用
        self.slt_item_num = 3
        self.combination_num = 6 # 3 4 6
        self.train_positive_samples = 50
        self.test_candidates_nums = 'all' # 'all' 100
        self.kcv = 5 # K交叉

        self.split_newScene_dataset_settings = (self.data_dir, self.num_negatives, self.slt_item_num, self.combination_num, self.train_positive_samples, self.test_candidates_nums, self.kcv)
        self.split_oldScene_dataset_settings = (self.data_dir, self.num_negatives, self.candidates_num, self.kcv)

        self.train_paras = (self.train_mode , self.train_new)
        # 区分结果
        self.learning_rates = self.CI_learning_rate,self.NI_learning_rate,self.topMLP_learning_rate
        self.train_name= 'lr:{}+{}+{}_l2:{}_{}_need_slt_apis:{}'.format(*self.learning_rates,self.l2_reg,self.train_mode,self.need_slt_apis)

        self.HIN_mode = HIN_mode
        # self.if_mashup_sem = if_mashup_sem
        # self.if_api_sem = if_api_sem
        # self.if_mashup_sim_only = if_mashup_sim_only
        # self.if_text_sem = True # 待改值
        # self.if_tag_sem = True
        # self.HIN_sim_paras = (self.if_mashup_sem,self.if_api_sem,self.if_mashup_sim_only,self.if_text_sem,self.if_tag_sem)
        self.new_HIN_paras = new_HIN_paras

        self.train_mashup_best = train_mashup_best
        """
        para = new_Para() # 加载时供其他类时使用，默认参数对象
        def set_current_para(p): # 初始化后的para需要赋值,不好用，其他类在import时指向了默认的para对象，不可变！！！
            global para
            para = p
        """

# 存储参数对象
class new_Para(object):
    param = new_Para_()  # 默认

    @classmethod
    def set_current_para(cls,p):
        cls.param = p

