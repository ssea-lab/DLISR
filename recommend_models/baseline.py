import heapq
import os
import sys
import random
sys.path.append("..")

from mf.get_UI import get_UV
from main.dataset import meta_data, dataset
from embedding.encoding_padding_texts import encoding_padding
from main.new_para_setting import new_Para
from main.evalute import evalute, summary

from gensim.corpora import Dictionary
from gensim.models import HdpModel, TfidfModel, LdaModel
import numpy as np
from Helpers.util import cos_sim

class gensim_data(object):
    def __init__(self,mashup_descriptions, api_descriptions, mashup_categories=None, api_categories=None,tag_times=2,mashup_only=False,strict_train=False):
        self.mashup_only =mashup_only
        self.strict_train = strict_train
        # 整合text和tag信息：一个mashup/api的信息整合在一起，一行
        if tag_times>0 and mashup_categories is not None:
            assert len(mashup_descriptions)==len(mashup_categories)
            self.mashup_dow=[[]]*len(mashup_descriptions)
            for i in range(len(mashup_descriptions)):
                self.mashup_dow[i]=mashup_descriptions[i]
                for j in range(tag_times):
                    self.mashup_dow[i] += mashup_categories[i] #  直接将文本和tag拼接，是否有更好的方法？增加出现次数？
        else:
            self.mashup_dow = mashup_descriptions
        self.mashup_dow = [[str (index) for index in indexes] for indexes in self.mashup_dow] # 二维列表
        # print (self.mashup_dow[0])

        if tag_times>0 and api_categories is not None:
            assert len (api_descriptions) == len (api_categories)
            self.api_dow=[[]]*len(api_descriptions)
            for i in range(len(api_descriptions)):
                self.api_dow[i]=api_descriptions[i]
                for j in range(tag_times):
                    self.api_dow[i]+=api_categories[i]
        else:
            self.api_dow=api_descriptions
        self.api_dow = [[str (index) for index in indexes] for indexes in self.api_dow]

        if not self.mashup_only and not self.strict_train:
            self.dct = Dictionary(self.mashup_dow + self.api_dow)
        if self.mashup_only and self.strict_train:
            # 训练用的mashup，api的编码
            self.train_mashup_dow = [self.mashup_dow[m_id] for m_id in dataset.crt_ds.his_mashup_ids]
            self.dct = Dictionary(self.train_mashup_dow)
            self.train_mashup_dow = [self.dct.doc2bow(mashup_info) for mashup_info in self.train_mashup_dow]  # 词id-数目
        # 无论怎样，总要为每个mashup/api计算feature
        self.mashup_dow = [self.dct.doc2bow(mashup_info) for mashup_info in self.mashup_dow]  # 所有mashup文本的词id-数目
        print('self.mashup_dow, num:',len(self.mashup_dow))
        zero_num = sum([1 if len(mashup_info)==0 else 0 for mashup_info in self.mashup_dow])
        print('zero_num',zero_num)
        self.api_dow = [self.dct.doc2bow(api_info) for api_info in self.api_dow]

        # print('len of self.mashup_dow,self.api_dow:{},{}'.format(len(self.mashup_dow),len (self.api_dow)))

        self.num_topics =0
        self.model = None # 处理文本的模型
        self._mashup_features= None # 文本提取的特征向量
        self._api_features= None

        self.mashup_topics = None # 文本最高的N个topic
        self.api_topics = None

    # 只关注词在文本中是否出现过，二进制，用于计算cos和jaccard
    def get_binary_v(self):
        dict_size=len(self.dct)
        mashup_binary_matrix=np.zeros((meta_data.mashup_num,dict_size))
        api_binary_matrix = np.zeros ((meta_data.api_num, dict_size))
        mashup_words_list=[] # 每个mashup中所有出现过的词
        api_words_list = []
        for i in range(meta_data.mashup_num):
            temp_words_list,_=zip(*self.mashup_dow[i])
            mashup_words_list.append(temp_words_list)
            for j in temp_words_list:# 出现的词汇index
                mashup_binary_matrix[i][j]=1.0

        for i in range(meta_data.api_num):
            temp_words_list,_=zip(*self.api_dow[i])
            api_words_list.append(temp_words_list)
            for j in temp_words_list:# 出现的词汇index
                api_binary_matrix[i][j]=1.0
        return mashup_binary_matrix,api_binary_matrix,mashup_words_list,api_words_list

    def model_pcs(self,model_name,LDA_topic_num=None):
        # hdp结果形式：[(0, 0.032271167132309014),(1, 0.02362695056720504)]
        if self.mashup_only:
            if self.strict_train:
                train_corpus = self.train_mashup_dow
            else:
                train_corpus = self.mashup_dow
        else:
            if self.strict_train:
                train_corpus = self.train_mashup_dow + self.train_api_dow
            else:
                train_corpus = self.mashup_dow + self.api_dow

        if model_name=='HDP':
            self.model = HdpModel(train_corpus, self.dct)
            self.num_topics = self.model.get_topics ().shape[0]
            print('num_topics',self.num_topics)
        elif model_name=='TF_IDF':
            self.model =TfidfModel (train_corpus)
            self.num_topics=len(self.dct)
        elif model_name=='LDA':
            if LDA_topic_num is None:
                self.model = LdaModel(train_corpus)
            else:
                self.model = LdaModel(train_corpus,num_topics=LDA_topic_num)
            self.num_topics = self.model.get_topics ().shape[0]

        else:
            raise ValueError('wrong gensim_model name!')

        # 使用模型处理文本，再转化为标准的np格式(每个topic上都有上)
        # print(self.mashup_dow)
        self.mashup_features=[self.model[mashup_info] for mashup_info in self.mashup_dow] # 每个mashup和api的feature
        # print(self.mashup_features)
        print('self.mashup_features, num:', len(self.mashup_features))
        zero_num1 = sum([1 if len(mashup_feature)==0 else 0 for mashup_feature in self.mashup_features])
        print('zero_num1',zero_num1)
        for i in range(len(self.mashup_features)):
            if len(self.mashup_features[i])==0:
                print(self.mashup_dow[i])

        self.api_features = [self.model[api_info] for api_info in self.api_dow]
        # print('when model-pcs,len of mashup_features and api_features:{},{}'.format(len(mashup_features),len(api_features)))
        self._mashup_features=np.zeros((meta_data.mashup_num, self.num_topics))
        self._api_features = np.zeros((meta_data.api_num, self.num_topics))
        for i in range(meta_data.mashup_num): # 部分维度有值，需要转化成规范array
            for index,value in self.mashup_features[i]:
                self._mashup_features[i][index]=value
        for i in range(meta_data.api_num):
            for index,value in self.api_features[i]:
                self._api_features[i][index]=value
        return self._mashup_features, self._api_features

    def get_topTopics(self,topTopicNum=3):# 选取概率最高的topK个主题 [(),(),...]
        mashup_topics = []
        api_topics = []
        for index in range(meta_data.mashup_num):
            sorted_mashup_feature = sorted(self.mashup_features[index],key = lambda x:x[1],reverse=True)
            try:
                topic_indexes,_ = zip(*sorted_mashup_feature)
            except:
                # 有时mashup_bow非空，但是mashup_feature为空
                topic_indexes = random.sample(range(meta_data.mashup_num),topTopicNum)
                # print(self.mashup_dow[index])
                # print(self.mashup_features[index])
                # print(sorted_mashup_feature)
                # raise ValueError('wrong 138!')
            num = min(len(topic_indexes),topTopicNum)
            mashup_topics.append(topic_indexes[:num])
        for index in range(meta_data.api_num):
            sorted_api_feature = sorted(self.api_features[index], key=lambda x: x[1], reverse=True)
            try:
                topic_indexes,_ = zip(*sorted_api_feature)
            except:
                topic_indexes = random.sample(range(meta_data.api_num), topTopicNum)
            num = min(len(topic_indexes),topTopicNum)
            api_topics.append(topic_indexes[:num])
        return mashup_topics,api_topics

def get_default_gd(default_encoding_texts=None,tag_times=2,mashup_only=False,strict_train=False): # 可传入encoding_texts对象
    if default_encoding_texts is None:
        default_encoding_texts = encoding_padding (meta_data.descriptions+meta_data.tags, True)
    mashup_descriptions=default_encoding_texts.texts_in_index_nopadding[:meta_data.mashup_num]
    api_descriptions   =default_encoding_texts.texts_in_index_nopadding[meta_data.mashup_num:meta_data.mashup_num+meta_data.api_num]
    mashup_categories  =default_encoding_texts.texts_in_index_nopadding[meta_data.mashup_num+meta_data.api_num:2*meta_data.mashup_num+meta_data.api_num]
    api_categories     =default_encoding_texts.texts_in_index_nopadding[2*meta_data.mashup_num+meta_data.api_num:]
    gd = gensim_data (mashup_descriptions, api_descriptions, mashup_categories, api_categories,tag_times,mashup_only=mashup_only,strict_train=strict_train) # 调整tag出现的次数
    return gd

# ***处理数据等最好不要放在recommend类中，并且该方法应设为recommend的子类？***
def Samanta(topK,if_pop=2,MF_mode='node2vec',pop_mode='',text_mode='HDP',LDA_topic_num=None):
    """
    :param Para:
    :param if_pop 如何使用pop  0 不使用；1，只做重排序；2总乘积做排序
    :param topK: 使用KNN表示新query的mf特征
    :param text_mode: 使用哪种特征提取方式  LDA  HDP
    :param pop_mode：pop值是否使用sigmoid规约到0-1区间
    :param pop_mode：MF_mode 为了省事，直接用node2vec得了
    :return:
    """

    api2pop=None
    if if_pop:
        api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs (pop_mode)

    root = os.path.join(dataset.crt_ds.root_path,'baselines')
    if not os.path.exists(root):
        os.makedirs(root)
    mashup_feature_path=os.path.join(root, 'mashup_{}.txt'.format(text_mode)) # ...
    api_feature_path = os.path.join(root, 'api_{}.txt'.format(text_mode))

    # 获取mashup_hdp_features,api_hdp_features
    if not os.path.exists(api_feature_path):
        gd=get_default_gd()
        _mashup_features,_api_features=gd.model_pcs(text_mode,LDA_topic_num)
        np.savetxt(mashup_feature_path,_mashup_features)
        np.savetxt(api_feature_path, _api_features)
    else:
        _mashup_features=np.loadtxt(mashup_feature_path)
        _api_features=np.loadtxt(api_feature_path)

    # Para.set_MF_mode(MF_mode) # 设置latent factor
    # new_Para.param.mf_mode = MF_mode # 修改参数对象，慎用

    candidate_ids_list = []
    all_predict_results=[]

    test_mashup_num = len(dataset.crt_ds.test_mashup_id_list)
    for i in range(test_mashup_num):
        test_mashup_id=dataset.crt_ds.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = dataset.crt_ds.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        # 用近邻mashup的latent factor加权表示自己
        localIndex2sim={}
        for local_index,train_m_id in enumerate(dataset.UV_obj.m_ids): # u_factors_matrix要用局部索引
            localIndex2sim[local_index]=cos_sim(_mashup_features[test_mashup_id],_mashup_features[train_m_id])
        topK_indexes,topK_sims=zip(*(sorted(localIndex2sim.items(), key=lambda x: x[1], reverse=True)[:topK]))
        topK_sims=np.array(topK_sims)/sum(topK_sims) # sim归一化
        cf_feature=np.zeros((new_Para.param.num_feat,))
        for z in range(len(topK_indexes)):
            cf_feature+= topK_sims[z] * dataset.UV_obj.m_embeddings[topK_indexes[z]]

        # 计算跟每个api的打分
        predict_results = []
        temp_predict_results=[] # 需要用pop进行重排序时的辅助
        api_zeros=np.zeros((new_Para.param.num_feat))
        for api_id in candidate_ids: # id
            a_id2index = dataset.UV_obj.a_id2index
            api_i_feature= dataset.UV_obj.a_embeddings[a_id2index[api_id]] if api_id in a_id2index.keys() else api_zeros  # 可能存在测试集中的api不在train中出现过的场景
            cf_score=np.sum(np.multiply(api_i_feature, cf_feature)) # mashup和api latent factor的内积
            sim_score=cos_sim(_mashup_features[test_mashup_id],_api_features[api_id]) # 特征的余弦相似度
            if if_pop==1:
                temp_predict_results.append((api_id,cf_score*sim_score))
            elif if_pop==0:
                predict_results.append(cf_score*sim_score)
            elif if_pop == 2:
                predict_results.append (cf_score * sim_score*api2pop[api_id])
        if if_pop==1:
            max_k_pairs = heapq.nlargest (topK, temp_predict_results, key=lambda x: x[1])  # 首先利用乘积排一次序
            max_k_candidates, _ = zip (*max_k_pairs)
            max_k_candidates=set(max_k_candidates)
            predict_results=[api2pop[api_id] if api_id in max_k_candidates else -1 for api_id in candidate_ids] # 重排序

        all_predict_results.append(predict_results)
    print('Samanta test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, dataset.crt_ds.grounds, new_Para.param.topKs)  # 评价
    _name='_pop_{}'.format(if_pop)
    _name+= new_Para.param.mf_mode
    csv_table_name = dataset.crt_ds.data_name + 'Samanta_model_{}'.format(topK)+_name + "\n"   # model.name
    summary(new_Para.param.evaluate_path, csv_table_name, evaluate_result, new_Para.param.topKs)  # 记录

    def divide(slt_apiNum):
        test_api_id_list_, predictions_, grounds_ = [], [], []
        for i in range(test_mashup_num):
            if len(dataset.crt_ds.slt_api_ids_instances[i]) == slt_apiNum:
                test_api_id_list_.append(candidate_ids_list[i])
                predictions_.append(all_predict_results[i])
                grounds_.append(dataset.crt_ds.grounds[i])
        return test_api_id_list_, predictions_, grounds_
    if new_Para.param.data_mode == 'newScene':
        for slt_apiNum in range(3):
            test_api_id_list_, predictions_, grounds_ = divide(slt_apiNum+1)
            evaluate_result = evalute(test_api_id_list_, predictions_, grounds_, new_Para.param.topKs)
            summary(new_Para.param.evaluate_path, str(slt_apiNum+1)+'_'+csv_table_name, evaluate_result, new_Para.param.topKs)  #



def hdp_pop(if_pop = True):
    # pop
    root = os.path.join(dataset.crt_ds.root_path,'baselines')
    if not os.path.exists(root):
        os.makedirs(root)
    mashup_hdp_path=os.path.join(root, 'mashup_HDP.txt') # ...
    api_hdp_path = os.path.join(root, 'api_HDP.txt')

    _mashup_hdp_features = np.loadtxt (mashup_hdp_path)
    _api_hdp_features = np.loadtxt (api_hdp_path)

    if if_pop:
        api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs ()
    # 测试
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(dataset.crt_ds.test_mashup_id_list)):
        test_mashup_id=dataset.crt_ds.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = dataset.crt_ds.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            sim_score=cos_sim(_mashup_hdp_features[test_mashup_id],_api_hdp_features[api_id])
            if if_pop:
                sim_score *= api2pop[api_id]
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('hdp_pop test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, dataset.crt_ds.grounds, new_Para.param.topKs)  # 评价
    name = 'hdp_pop' if if_pop else 'hdp'
    csv_table_name = dataset.crt_ds.data_name + name + "\n"   # model.name
    summary(new_Para.param.evaluate_path, csv_table_name, evaluate_result, new_Para.param.topKs)  # 记录


def TF_IDF(if_pop):
    """
    可以跟写到Samanta的类中，但太混乱，没必要
    :return:
    """
    gd = get_default_gd()
    api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs()
    _mashup_IFIDF_features, _api_IFIDF_features = gd.model_pcs ('TF_IDF')

    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(dataset.crt_ds.test_mashup_id_list)):
        test_mashup_id=dataset.crt_ds.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = dataset.crt_ds.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            sim_score=cos_sim(_mashup_IFIDF_features[test_mashup_id],_api_IFIDF_features[api_id])
            if if_pop:
                predict_results.append(sim_score*api2pop[api_id])
            else:
                predict_results.append(sim_score )
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('TF_IDF test,done!')

    name = 'TFIDF_pop' if if_pop else 'TFIDF'
    evaluate_result = evalute(candidate_ids_list, all_predict_results, dataset.crt_ds.grounds, new_Para.param.topKs)  # 评价
    csv_table_name = dataset.crt_ds.data_name + name + "\n"   # model.name
    summary(new_Para.param.evaluate_path, csv_table_name, evaluate_result, new_Para.param.topKs)  # 记录

def MF(train_datas,test_datas,mode = ''):
    all_predict_results=[] # 每个测试样例(多个api的)的评分
    for slt_num in range(1,new_Para.param.slt_item_num+1): # 不同个数的训练测试集
        test_mashup_id_list, test_api_id_list, grounds = test_datas[slt_num-1]
        # 增加处理和读取MF结果的接口
        UV_obj = get_UV(dataset.crt_ds.root_path, mode,train_datas[slt_num-1],slt_num)
        m_id2index,a_id2index = UV_obj.m_id2index,UV_obj.a_id2index
        for i in range(len(test_mashup_id_list)):
            test_mashup_id=test_mashup_id_list[i][0] # 每个mashup id
            predict_results = []
            for test_api_id in test_api_id_list[i]: # id
                if test_mashup_id not in m_id2index or test_api_id not in a_id2index:
                    dot = 0
                else:
                    m_embedding = UV_obj.m_embeddings[m_id2index[test_mashup_id]]
                    a_embedding = UV_obj.a_embeddings[a_id2index[test_api_id]]
                    dot = np.dot(m_embedding,a_embedding)
                predict_results.append(dot)
            all_predict_results.append(predict_results)
        print('{}_{} test,done!'.format(mode,slt_num))

        evaluate_result = evalute(test_api_id_list, all_predict_results, dataset.crt_ds.grounds, new_Para.param.topKs)  # 评价
        csv_table_name = dataset.crt_ds.data_name + mode + str(slt_num)+ "\n"   # model.name
        summary(new_Para.param.evaluate_path, csv_table_name, evaluate_result, new_Para.param.topKs)  # 记录

def pop():
    """
    :return:
    """
    api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs ()
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(dataset.crt_ds.test_mashup_id_list)):
        test_mashup_id=dataset.crt_ds.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = dataset.crt_ds.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            predict_results.append(api2pop[api_id])
        all_predict_results.append(predict_results)
    print('pop test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, dataset.crt_ds.grounds, new_Para.param.topKs)  # 评价
    csv_table_name = dataset.crt_ds.data_name + 'pop' + "\n"   # model.name
    summary(new_Para.param.evaluate_path, csv_table_name, evaluate_result, new_Para.param.topKs)  # 记录

# """service package recommendation for mashup creation via mashup textual description mining"""


# “a novel approach for API recommendation in mashup development”
def binary_keyword(if_pop = False):
    # pop
    api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs ()
    gd = get_default_gd()
    mashup_binary_matrix, api_binary_matrix, mashup_words_list, api_words_list = gd.get_binary_v ()


    # 测试WVSM(Weighted Vector Space Model)
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(dataset.crt_ds.test_mashup_id_list)):
        test_mashup_id=dataset.crt_ds.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = dataset.crt_ds.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            if if_pop:
                sim_score = cos_sim(mashup_binary_matrix[test_mashup_id], api_binary_matrix[api_id]) * api2pop[api_id]
            else:
                sim_score = cos_sim(mashup_binary_matrix[test_mashup_id], api_binary_matrix[api_id]) # 测试只使用特征向量的效果
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('WVSM test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, dataset.crt_ds.grounds, new_Para.param.topKs)  # 评价
    name = 'WVSM_pop' if if_pop else 'WVSM'
    csv_table_name = dataset.crt_ds.data_name + name + "\n"   # model.name
    summary(new_Para.param.evaluate_path, csv_table_name, evaluate_result, new_Para.param.topKs)  # 记录

    """
    # 测试WJaccard(Weighted Jaccard)
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(dataset.crt_ds.test_mashup_id_list)):
        test_mashup_id=dataset.crt_ds.test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = dataset.crt_ds.test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            mashup_set=set(mashup_words_list[test_mashup_id])
            api_set = set (api_words_list[api_id])
            if if_pop:
                sim_score=1.0*len(mashup_set.intersection(api_set))/len(mashup_set.union(api_set))*api2pop[api_id]
            else:
                sim_score = 1.0 * len(mashup_set.intersection(api_set)) / len(mashup_set.union(api_set))
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('WJaccard test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, dataset.crt_ds.grounds, new_Para.param.topKs)  # 评价
    name = 'WJaccard_pop' if if_pop else 'WJaccard'
    csv_table_name = dataset.crt_ds.data_name + name + "\n"   # model.name
    summary(new_Para.param.evaluate_path, csv_table_name, evaluate_result, new_Para.param.topKs)  # 记录
    """



if __name__=='__main__':
    # Samanta(topK, if_pop=1, MF_mode='pmf', pop_mode='')
    for mf in ['BPR', 'pmf', 'nmf', 'listRank']:  # 'pmf',
        for k in [10,20,30,40,50]: # ,100
            for if_pop in [1,2]:
                for pop_mode in ['']:# ,'sigmoid'
                    print('{},{},{},{}:'.format(k,if_pop,mf,pop_mode))
                    Samanta(k,if_pop=if_pop,MF_mode=mf,pop_mode=pop_mode)
    """"""
    # Samanta (50, if_pop=1, MF_mode='BPR') # 效果最好
    # TF_IDF()
    # binary_keyword ()  # 效果好的不敢相信。。相比之下我们的算法只提高了10%
    # pop()

    # hdp_pop () # Samanta使用pop*sim
