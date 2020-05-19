import os
import pickle
import sys
sys.path.append("..")
from embedding.embedding import get_embedding_matrix
from embedding.encoding_padding_texts import encoding_padding
from main.dataset import meta_data, dataset
from main.evalute import evalute_by_epoch
from main.new_para_setting import new_Para
from recommend_models.HIN_sim import mashup_HIN_sims
from recommend_models.baseline import get_default_gd
from random import choice
import numpy as np
import math

def sigmoid(x):
    return 1.0/(1+math.exp(-x))


class HINRec_new(object):
    def __init__(self,model_name = 'PasRec',semantic_mode='HDP',LDA_topic_num=None,epoch_num=15,neighbor_size=15,topTopicNum=3,cluster_mode='LDA',cluster_mode_topic_num=100):
        # topTopicNum在PasRec中用于计算content相似度；在IsRec中用于从K个类中寻找近邻

        # semantic_mode='HDP',LDA_topic_num=None: about feature in HIN  只在IsRec_best中使用，因为PasRec和IsRec计算文本相似度时要么使用topic作为tag，要么使用EmbMax
        # cluster_mode='LDA',cluster_mode_topic_num: ABOUT clustering by LDA...
        self.simple_name = model_name
        self.epoch_num = epoch_num
        self.neighbor_size = neighbor_size # 找最近邻时的规模
        self.topTopicNum = topTopicNum
        if self.simple_name == 'IsRec_best':
            self.p1_weight, self.p2_weight, self.p3_weight = 1/3,1/3,1/3
            self.path_weights = [self.p1_weight, self.p2_weight, self.p3_weight]
        elif self.simple_name == 'PasRec_2path':
            self.p1_weight, self.p2_weight = 1/2,1/2
            self.path_weights = [self.p1_weight, self.p2_weight]
        elif self.simple_name == 'IsRec':
            self.p1_weight,self.p2_weight,self.p3_weight,self.p4_weight,self.p5_weight,self.p6_weight,self.p7_weight = 1/7,1/7,1/7,1/7,1/7,1/7,1/7
            self.path_weights = [self.p1_weight,self.p2_weight,self.p3_weight,self.p4_weight,self.p5_weight,self.p6_weight,self.p7_weight]
        else :
            self.p1_weight,self.p2_weight,self.p3_weight,self.p4_weight,self.p5_weight,self.p6_weight = 1/6,1/6,1/6,1/6,1/6,1/6
            self.path_weights = [self.p1_weight,self.p2_weight,self.p3_weight,self.p4_weight,self.p5_weight,self.p6_weight]


        self.learning_rate = 0.001
        self.reg=0.001
        # 'new_true'  _25pairs

        if LDA_topic_num is None:
            LDA_topic_num = ''
        self.model_name = '{}_{}_epoch{}_nbSize{}TopicNum{}{}{}NEW'.format(model_name,semantic_mode, epoch_num, neighbor_size,topTopicNum,cluster_mode,cluster_mode_topic_num)
        self.model_dir = dataset.crt_ds.model_path.format(self.model_name) # 模型路径 # !!!
        self.weight_path = os.path.join(self.model_dir, 'weights.npy')  # 最核心的数据，只保存它，其他无用！

        # 数据集相关
        self.all_mashup_num = meta_data.mashup_num
        self.all_api_num = meta_data.api_num
        self.his_m_ids = dataset.crt_ds.his_mashup_ids
        self.his_m_ids_set = set(self.his_m_ids)

        # 没区分
        # self.train_mashup_api_list = meta_data.mashup_api_list # 纯正例的训练集!!!
        # self.train_mashup_api_dict = meta_data.pd.get_mashup_api_pair('dict')

        # 严格的训练集！！！
        self.train_mashup_api_list = [pair for pair in meta_data.mashup_api_list if pair[0] in self.his_m_ids_set]
        self.train_mashup_api_dict = {key: value for key,value in meta_data.pd.get_mashup_api_pair('dict').items() if key in self.his_m_ids_set}
        print(len(self.train_mashup_api_dict))

        # 训练数据集 api_id: set(mashup_ids)
        self.train_aid2mids = {}
        for mashup_id, api_id in self.train_mashup_api_list:
            if api_id not in self.train_aid2mids.keys():
                self.train_aid2mids[api_id] = set()
            self.train_aid2mids[api_id].add(mashup_id)
        self.his_a_ids = list(self.train_aid2mids.keys())  # 训练数据集中出现的api_id !!!
        self.notInvokeScore = 0 # 加入评价的api是历史mashup从未调用过的，基准评分0.5；参考1和0  0.5很差！！！

        # 文本，HIN相似度相关
        self.HIN_path = os.path.join(self.model_dir, 'HIN_sims') # 存储各个HIN_sim源文件的root !!!
        self.semantic_mode = semantic_mode
        self.LDA_topic_num = LDA_topic_num
        encoded_texts = encoding_padding (meta_data.descriptions + meta_data.tags, new_Para.param.remove_punctuation) # 文本编码对象
        embedding_matrix = get_embedding_matrix(encoded_texts.word2index, new_Para.param.embedding_name,dimension=new_Para.param.embedding_dim) # 每个编码词的embedding

        # HIN中 文本相似度计算  只在IsRec_best中使用，因为PasRec和IsRec计算文本相似度时要么使用topic作为tag，要么使用EmbMax!!!
        HIN_gd = get_default_gd(encoded_texts,tag_times=0,mashup_only=True,strict_train=True) # 用gensim处理文本,文本中不加tag
        self._mashup_features ,self._api_features = HIN_gd.model_pcs(self.semantic_mode ,self.LDA_topic_num) # IsRec_best需要使用TF_IDF!!!
        features = self._mashup_features, self._api_features
        self.mhs = mashup_HIN_sims(embedding_matrix, encoded_texts, semantic_name=self.semantic_mode,HIN_path=self.HIN_path,features=features,if_text_sem=True,if_tag_sem=False) # 计算HIN_sim的对象,传入的是mashup和api的文本feature
        self.mID2PathSims={} # 每个mashupID(含已调用apis)，跟历史mashup的各种路径的相似度
        self.HIN_sims_changed_flag = False

        # topTopicNum在PasRec中用于基于LDA等的主题计算content相似度；在IsRec中用于从K个类中寻找近邻!!!
        topic_gd = get_default_gd(encoded_texts,tag_times=0,mashup_only=True,strict_train=True) # 用gensim处理文本,文本中不加tag
        topic_gd.model_pcs(cluster_mode,cluster_mode_topic_num) # 暂时用HDP分类/提取特征;确定主题数之后改成LDA
        self.m_id2topic,self.a_id2topic = topic_gd.get_topTopics(self.topTopicNum)

        self.topic2m_ids = {} # topic到mashup的映射；相当于按主题分类 全部mashup！不区分训练集测试集！
        for m_id,topic_indexes in enumerate(self.m_id2topic):
            for topic_index in topic_indexes:
                if topic_index not in self.topic2m_ids:
                    self.topic2m_ids[topic_index] = []
                self.topic2m_ids[topic_index].append(m_id)

        self.read_model() # 主要读取权重参数，其他不重要

    # 该模型在训练时，训练样本的相似度文件临时存储在self.mID2PathSims中，不需永久存储，一次训练即可
    # 为NI服务时，训练和测试样本的相似度文件都需要永久存储；包括各种path的相似度和某一种权重组合的相似度，以及特征

    # 计算一个mashup(可能需要已选择服务)到其他mashup的各种相似度，可选择是否存入到self.mID2PathSims中(NI的实例调用时)
    def get_id2PathSims(self,m_id,slt_apis_list=None,if_temp_save=True,if_cutByTopics=True):
        key = (m_id,tuple(slt_apis_list)) if slt_apis_list is not None else m_id
        if key in self.mID2PathSims.keys(): # 重新加载该模型时为空，有必要时为NI即时计算
            return self.mID2PathSims.get(key)
        else:
            his_m_ids = self.his_m_ids
            # IsRec是否使用剪枝策略：拥有相同tag的所有mashup中选择近邻
            if 'IsRec' in self.simple_name and if_cutByTopics:
                his_m_ids = []
                for topic in self.m_id2topic[m_id]:
                    sameTopic_his_mids = [sameTopic_mid for sameTopic_mid in self.topic2m_ids[topic] if
                                          sameTopic_mid in self.his_m_ids]  # 一定要是训练集中的mashup！！！
                    his_m_ids += sameTopic_his_mids
            if self.simple_name == 'PasRec':
                id2P1Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                # 这里较为特殊：使用content的topic作为tag，用get_p1_sim
                id2P2Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath', self.m_id2topic) for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P3Sim = {neigh_m_id: self.mhs.get_p3_sim(neigh_m_id, slt_apis_list) for neigh_m_id in his_m_ids if  neigh_m_id != m_id}
                id2P4Sim = {neigh_m_id: self.mhs.get_p4_sim(neigh_m_id, slt_apis_list, 'MetaPath') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                # 这里较为特殊：使用content的topic作为tag，用get_p4_sim
                id2P5Sim = {neigh_m_id: self.mhs.get_p4_sim(neigh_m_id, slt_apis_list, 'MetaPath', self.a_id2topic) for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P6Sim = {neigh_m_id: self.mhs.get_p6_sim(neigh_m_id, slt_apis_list) for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2PathSims = [id2P1Sim, id2P2Sim, id2P3Sim, id2P4Sim, id2P5Sim, id2P6Sim]  #
            elif self.simple_name == 'IsRec':
                id2P1Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P2Sim = {neigh_m_id: self.mhs.get_p2_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'EmbMax') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P3Sim = {neigh_m_id: self.mhs.get_p3_sim(neigh_m_id, slt_apis_list) for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P4Sim = {neigh_m_id: self.mhs.get_p4_sim(neigh_m_id, slt_apis_list, 'MetaPath') for neigh_m_id in  his_m_ids if neigh_m_id != m_id}
                id2P5Sim = {neigh_m_id: self.mhs.get_p5_sim(neigh_m_id, slt_apis_list, 'EmbMax') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P6Sim = {neigh_m_id: self.mhs.get_p6_sim(neigh_m_id, slt_apis_list) for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P7Sim = {neigh_m_id: self.mhs.get_p2_sim_sem(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'TF_IDF') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2PathSims = [id2P1Sim, id2P2Sim, id2P3Sim, id2P4Sim, id2P5Sim, id2P6Sim,id2P7Sim]  #
            elif self.simple_name == 'PasRec_2path':
                id2P1Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                # 这里较为特殊：使用content的topic作为tag，用get_p1_sim
                id2P2Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath', self.m_id2topic) for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2PathSims = [id2P1Sim, id2P2Sim]  #
            elif self.simple_name == 'IsRec_best':
                id2P1Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P2Sim = {neigh_m_id: self.mhs.get_p2_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'EmbMax') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2P3Sim = {neigh_m_id: self.mhs.get_p2_sim_sem(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'TF_IDF') for neigh_m_id in his_m_ids if neigh_m_id != m_id}
                id2PathSims = [id2P1Sim, id2P2Sim, id2P3Sim]  #
            if if_temp_save:
                self.mID2PathSims[key] = id2PathSims
            return id2PathSims

    def predict_an_instance(self,m_id,a_id,slt_apis_list,if_score_only = False):
        # 根据当前的某种路径的相似度，预测当前mashup(在已选择slt_apis_list的情况下)对api的一种评分;通用
        def get_path_score(m_id, a_id, neighborId2sim):
            if a_id not in self.train_aid2mids.keys(): # 没有被调用过的服务，user-based的机制，评分肯定为0
                score_sum = self.notInvokeScore
            else:
                # 计算某种路径下的score/sim，要输入一个mashup和每个历史mashup的某种路径下的相似度
                num= min(self.neighbor_size, len(neighborId2sim))
                sorted_id2sim = sorted(neighborId2sim.items(), key=lambda x:x[1], reverse=True) [:num]
                score_sum = 0
                for neighbor_m_id,temp_sim in sorted_id2sim: # 最相似的近邻 论文中提到还需要是调用过api的mashup，这里通过分值为0可以达到同样目的
                    if neighbor_m_id in self.train_aid2mids[a_id]:
                        temp_score = 1
                    else:
                        temp_score = 0
                    score_sum += temp_score*temp_sim
            return score_sum

        id2PathSims = self.get_id2PathSims(m_id,slt_apis_list) # 计算某个实例的pathSims，跟weight无关，计算一次不会变
        path_scores = [get_path_score(m_id,a_id,id2PathSim) for id2PathSim in id2PathSims] # 几种路径下的score,更新模型时有用
        score = sum(np.array(path_scores)*np.array(self.path_weights))
        if if_score_only:
            return score
        else:
            return score,path_scores

    def get_true_candi_apis(self):
        self.mid2candiAids = {}
        self.mid2candiAids_path = os.path.join(self.model_dir, 'true_candi_apis.txt') # 根据IsRec的思想，只把近邻mashup调用过的服务作为候选
        def save_dict():
            with open(self.mid2candiAids_path,'w') as f:
                for m_id, true_candi_apis in self.mid2candiAids.items():
                    f.write(str(m_id)+' '+' '.join([str (id) for id in true_candi_apis])) #
                    f.write('\n')
        def read_dict():
            with open(self.mid2candiAids_path,'r') as f:
                line = f.readline()
                while line is not None:
                    ids = [int(str_id) for str_id in line.split()]
                    if len(ids) == 0:
                        break
                    self.mid2candiAids[ids[0]] = ids[1:]
                    line = f.readline()
        if not os.path.exists(self.mid2candiAids_path):
            for key,id2PathSims in self.mID2PathSims.items():
                m_id = key[0] # [0] ??? # key = (m_id,tuple(slt_apis_list))
                if m_id not in self.mid2candiAids.keys():
                    all_neighbor_mids = set()
                    for id2sim in id2PathSims: # 到各个剪枝后的候选近邻的某种路径下的相似度
                        num= min(self.neighbor_size,len(id2sim))
                        sorted_id2sim = sorted(id2sim.items(),key=lambda x:x[1],reverse=True) [:num] # 某种路径下的近邻
                        all_neighbor_mids = all_neighbor_mids.union({id for id,sim in sorted_id2sim})
                    true_candi_apis = set()
                    for neighbor_mid in all_neighbor_mids:
                        if neighbor_mid not in self.train_mashup_api_dict.keys():
                            print(neighbor_mid)
                            print(neighbor_mid in self.his_m_ids)
                        true_candi_apis = true_candi_apis.union(set(self.train_mashup_api_dict[neighbor_mid])) # 该近邻mashup调用过的api # ???
                    self.mid2candiAids[m_id] = true_candi_apis
            save_dict()
        else:
            read_dict()
        return self.mid2candiAids

    def predict(self,args,verbose=0): # 仿照DL的model,返回多个实例的评分
        m_ids, a_ids, slt_apis_lists = args[0],args[1],args[2]
        num = len(m_ids)
        if not slt_apis_lists:
            predictions = [self.predict_an_instance(m_ids[i], a_ids[i], None, if_score_only=True) for i in range(num)]
        else:
            predictions = [self.predict_an_instance(m_ids[i], a_ids[i], slt_apis_lists[i],if_score_only = True) for i in range(num)]
        return np.array(predictions)

    def get_instances(self,mashup_id_instances, api_id_instances, slt_api_ids_instances=None):
        # 不需要array，不需要填充，输出结果可以使用predict就好;供evalute_by_epoch使用
        return (mashup_id_instances, api_id_instances, slt_api_ids_instances)

    def train(self,test_data):
        """
        每20次测试一次，训练数据不用输入，用meta_data获取
        :param test_data:
        :return:
        """
        for index in range(self.epoch_num):
            loss = 0
            # 模仿librec的实现，每个api跟一对正负mashup组成一个样例，每个api的样本数最大为50；(均衡性问题？）
            for sampleCount in range(len(self.his_a_ids) * 50):  # 每个  # 之前是50
                while (True):
                    a_id = choice(self.his_a_ids)
                    if len(self.train_aid2mids[a_id]) == len(self.his_m_ids):  # 如果被所有mashup调用，则没有负例
                        continue
                    pos_m_ids = self.train_aid2mids[a_id]  # 正例
                    pos_m_id = choice(list(pos_m_ids))
                    neg_m_ids = self.his_m_ids_set - pos_m_ids
                    neg_m_id = choice(list(neg_m_ids))
                    break

                # 训练时计算相似度，已选择的服务应该不包含当前服务！！！
                posPredictRating,posPathScores = self.predict_an_instance(pos_m_id, a_id, self.train_mashup_api_dict[pos_m_id]-{a_id})
                negPredictRating,negPathScores = self.predict_an_instance(neg_m_id, a_id, self.train_mashup_api_dict[neg_m_id]-{a_id})
                diffValue = posPredictRating - negPredictRating
                deriValue = sigmoid(-diffValue);
                lossValue = -math.log(sigmoid(diffValue))
                loss += lossValue

                for i in range(len(self.path_weights)): # 第i条路径对应的参数
                    temp_value = self.path_weights[i]
                    self.path_weights[i] += self.learning_rate* (deriValue * (posPathScores[i]-negPathScores[i]) - self.reg * temp_value)
                    loss += self.reg * temp_value * temp_value
            print('epoch:{}, loss:{}'.format(index, loss))

            if index>0 and index%20==0:
                self.test_model(test_data)

    def test_model(self,test_data):
        evalute_by_epoch(self, self, self.model_name, test_data)

    def save_model(self):
        # 存储HIN相似度文件和参数权重
        np.save(self.weight_path ,np.array(self.path_weights))
        print('save weights,done!')

    def read_model(self):
        if os.path.exists(self.weight_path):
            self.path_weights = np.load(self.weight_path)
            print('read weights,done!')
