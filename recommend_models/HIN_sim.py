import os
import pickle
import sys
sys.path.append("..")


import numpy as np
from main.new_para_setting import new_Para
from main.dataset import meta_data, dataset
from Helpers.util import cos_sim
from main.processing_data import get_mashup_api_allCategories, get_mashup_api_field


def cpt_p13_sim(s_list1,s_list2):
    """
    输入每个mashup的servcie构成或标签序列，计算SBS-category-SBS，SBS-service-SBS相似度
    :param s_list1:
    :param s_list2:
    :return:
    """

    set1,set2=set(s_list1),set(s_list2)
    sum_len = len(set1)+len(set2)
    return 0  if sum_len == 0  else 2*len(set1&set2)/sum_len#  乘以2！！！


def cpt_p46_sim(dict1,dict2):
    """
    SBS-service-category-service-SBS    SBS-service-provider-service-SBS
    :param dict1: key：service, value: services 对应的categorys/provider
    :param dict2:
    :return:
    """
    num_dict1_values=sum([len(values) for values in dict1.values()])
    num_dict2_values = sum ([len (values) for values in dict2.values ()])
    denom = num_dict1_values + num_dict2_values
    if denom==0:
        return 0
    else:
        return sum ([len (dict1[key]) for key in dict1.keys()&dict2.keys ()])/denom # 公用的service所包含的category/provider之和

class word_sim(object):
    def __init__(self,wordindex2emb):
        """
        wordindex2emb 是encode之后的index 到 embedding的映射，由模型的embedding层参数初始化
        :param wordindex2emb:
        """
        self.words_Sim={} # 词汇词汇之间的相似度
        self.wordindex2embedding=wordindex2emb

    def get_word_cos_sim(self, id1, id2):
        """
        计算词（id）间的sim，并存储供索引
        :param id1:
        :param id2:
        :return:
        """
        if id1==0 or id2==0: # 是padding用的0 index时，返回索引
            return 0
        if id1 == id2:
            return 1
        id_b = max (id1, id2)
        id_s = min (id1, id2)
        value = self.words_Sim.get ((id_s, id_b))  # 小到大，按顺序
        if value is None:
            value = cos_sim (self.wordindex2embedding[id_s], self.wordindex2embedding[id_b])
            self.words_Sim[(id_s, id_b)] = value
        return value

def cpt_2list_sim(sim_matrix):
    # 通用的计算两个集合相似度的方式，需要输入相似度矩阵
    max_sum1 = np.sum (np.max (sim_matrix, axis=0))
    max_sum2 = np.sum (np.max (sim_matrix, axis=1))
    return (max_sum1 + max_sum2) / (len (sim_matrix) + len (sim_matrix[0]))

def cpt_content_sim(ws,word_list1, word_list2):
    # word_list1 是encode过的index形式
    sim_matrix = np.array ([[ws.get_word_cos_sim (id1, id2) for id2 in word_list2] for id1 in word_list1])  # 2d
    return cpt_2list_sim(sim_matrix)

    #@@@@！！！！！@@@@@
    # 对象对双语义的方法很难搞！！！之前求过还可用，
    # 如果之前没有存储过，需要求，但是使用的特征不是同一种类型的。所以考虑把特征做出方法的参数传进去？？？
class mashup_HIN_sims(object):
    def __init__(self, wordindex2emb,encoded_texts,HIN_path='',features=None,semantic_name='',if_text_sem = True,if_tag_sem=True,if_mashup_sem=True,if_api_sem=True):

        self.ws = word_sim (wordindex2emb)# embedding 层参数
        self.encoded_texts = encoded_texts
        self.num_users = meta_data.mashup_num
        self.num_items = meta_data.api_num
        self.semantic_name= semantic_name # 输入的特征的名字  默认为空，是利用CI模型做的；其他的使用HDP等要输入
        # 下一步可以把以下几个对象设为encoded_texts的属性，因为baseline算法中也要使用
        self.unpadded_encoded_mashup_texts = self.encoded_texts.texts_in_index_nopadding[:self.num_users]
        self.unpadded_encoded_api_texts = self.encoded_texts.texts_in_index_nopadding[self.num_users:(self.num_users + self.num_items)]
        self.unpadded_encoded_mashup_tags = self.encoded_texts.texts_in_index_nopadding[(self.num_users + self.num_items):(2 * self.num_users + self.num_items)]
        self.unpadded_encoded_api_tags = self.encoded_texts.texts_in_index_nopadding[(2 * self.num_users + self.num_items):]

        # 利用content和tag的feature计算相似度
        if features is not None:
            if len(features) ==2:
                if if_text_sem and not if_tag_sem: # 可以只用text的语义特征,PasRec等使用
                    self.mashup_texts_features, self.api_texts_features = features
                if if_mashup_sem and not if_api_sem: # 可以只用mashup的语义特征  HIN中使用
                    self.mashup_texts_features, self.mashup_tag_features = features
            elif len(features) ==4:
                self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features=features

        self.mashup_id2info = meta_data.mashup_id2info
        self.api_id2info = meta_data.api_id2info
        self.mashup_apis = meta_data.pd.get_mashup_api_pair ('dict')
        self.api_id2provider = [get_mashup_api_field(self.api_id2info, a_id, 'API Provider') for a_id in range(self.num_items)]

        # self.path= os.path.join(HIN_path,self.name) # 存放在CI文件夹下！dataset.crt_ds.root_path  no_kcv_root_path
        self.path = HIN_path # 存放相似度的路径  kcvIndex/CIModelPath/HIN_sims/

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.p1_sims,self.p2_sims,self.p3_sims,self.p4_sims,self.p5_sims,self.p6_sims = None,None,None,None,None,None
        self.p1_sims_sem,self.p2_sims_sem,self.p3_sims_sem,self.p4_sims_sem = None,None,None,None

        # 传递引用，sims本身也是引用，sims= load_sims('p{}_sims.dat'也不行！！！
        # for i,sims in enumerate([self.p1_sims, self.p2_sims,self.p3_sims, self.p4_sims, self.p5_sims, self.p6_sims]):
        #     load_sims('p{}_sims.dat'.format(i+1), sims)
        #
        # for i,sims_sem in zip([1,2,4,5],[self.p1_sims_sem, self.p2_sims_sem, self.p4_sims_sem, self.p5_sims_sem]):
        #     load_sims('p{}_sims_sem.dat'.format(i), sims_sem)

        # print('after,p1_sims_sem{}.dat'.format(self.semantic_name), len(self.p1_sims_sem))

        # 判断sim文件是否发生修改
        self.flag1,self.flag2,self.flag3,self.flag4,self.flag5,self.flag6=False,False,False,False,False,False
        self.flag1_sem, self.flag2_sem, self.flag4_sem = False,False,False

    def load_sims(self,name):
        # 如果之前有，读取
        sim_path = os.path.join(self.path, name)
        if os.path.exists(sim_path):
            with open(sim_path, 'rb') as f:
                sims = pickle.load(f)  # 先读取
                return sims
                # print('load {},done!'.format(name))
                # print('{},size:{}'.format(i + 1, len(sims)))
        else:
            return {}

    # def get_p1_sim(self,min_m_id, max_m_id,if_semantic,if_tag_sem,mashup_tags = None):
    #     """
    #     # p1:SBS-category-SBS
    #     :param min_m_id:
    #     :param max_m_id:
    #     :param self_sim_dict: 传入self.p1_sim 或是self.p1_sim_sem
    #     :param if_semantic: 基于feature+cosine 还是基于meta-path
    #     :param api_tags 自己传入的类别标签（PasRec中计算content相似度时使用topic也可以用这种）！！！
    #     :return:
    #     """
    #     self_sim_dict = self.p1_sims_sem if if_semantic else self.p1_sims
    #     # print('sim_dict1,size:',len(self_sim_dict))
    #     # print('p1_sims_sem,size:',len(self.p1_sims_sem))
    #     if (min_m_id, max_m_id) not in self_sim_dict.keys():
    #         # print(min_m_id, max_m_id,'not in p1 sim!')
    #         if if_semantic and if_tag_sem: # 使用tag语义时
    #             self.flag1_sem = True
    #             p1_sim = cos_sim(self.mashup_tag_features[min_m_id],self.mashup_tag_features[max_m_id])
    #         else:
    #             self.flag1=True
    #             if mashup_tags is None: # 也可以使用用户自己传入的tag
    #                 mashup_tags = self.unpadded_encoded_mashup_tags
    #
    #             # 直接存储mashup到tag矩阵，省去调用搜索时间？ 对tag编码，字符串->int 加速操作
    #             mashup_categories1 = mashup_tags[min_m_id]
    #             mashup_categories2 = mashup_tags[max_m_id]
    #             p1_sim = cpt_p13_sim (mashup_categories1, mashup_categories2)
    #
    #         self_sim_dict[(min_m_id, max_m_id)] = p1_sim
    #     else:
    #         p1_sim=self_sim_dict[(min_m_id, max_m_id)]
    #     return p1_sim
    #
    # def get_p2_sim(self,min_m_id, max_m_id,if_semantic,if_text_sem):
    #     # p2:SBS-content-SBS
    #     # 基于feature+cosine还是词汇embedding集合最大匹配
    #     self_sim_dict = self.p2_sims_sem if if_semantic else self.p2_sims
    #     if (min_m_id, max_m_id) not in self_sim_dict.keys ():
    #         # print('min_m_id, max_m_id,not in p2 sim!')
    #         if if_semantic and if_text_sem:
    #             self.flag2_sem = True
    #             p2_sim = cos_sim(self.mashup_texts_features[min_m_id],self.mashup_texts_features[max_m_id])
    #         else:
    #             self.flag2 = True
    #             # 1.编码过的文本索引也存储起来 2.不用每次编码，不需要加0，增加工作量...
    #             p2_sim = cpt_content_sim (self.ws, self.unpadded_encoded_mashup_texts[min_m_id], self.unpadded_encoded_mashup_texts[max_m_id])
    #         self_sim_dict[(min_m_id, max_m_id)] = p2_sim
    #     else:
    #         p2_sim = self_sim_dict[(min_m_id, max_m_id)]
    #     return p2_sim

    def get_p1_sim(self,min_m_id, max_m_id,mTagFalseSem=None,mashup_tags = None):
        """
        # p1:SBS-category-SBS 非语义形式
        :param min_m_id:
        :param max_m_id:
        :param mTagTrueSem: tag的“非语义形式”:'EmbMax' 'MetaPath'
        :param mashup_tags 自己传入的类别标签（PasRec中计算content相似度时使用topic也可以用这种）！！！
        :return:
        """
        if mTagFalseSem is not None:
            if self.p1_sims is None:
                self.p1_sims = self.load_sims('p1_sims{}.dat'.format(mTagFalseSem))
            if (min_m_id, max_m_id) not in self.p1_sims.keys():
                # print(min_m_id, max_m_id,'not in p1 sim!')
                self.flag1=True
                if mashup_tags is None: # 也可以使用用户自己传入的tag
                    mashup_tags = self.unpadded_encoded_mashup_tags

                # 直接存储mashup到tag矩阵，省去调用搜索时间？ 对tag编码，字符串->int 加速操作
                mashup_categories1 = mashup_tags[min_m_id]
                mashup_categories2 = mashup_tags[max_m_id]
                if mTagFalseSem=='MetaPath':
                    p1_sim = cpt_p13_sim (mashup_categories1, mashup_categories2)
                elif mTagFalseSem=='EmbMax':
                    p1_sim = cpt_content_sim (self.ws,mashup_categories1, mashup_categories2) # 对tag采用embedding的方法
                self.p1_sims[(min_m_id, max_m_id)] = p1_sim
            else:
                p1_sim=self.p1_sims[(min_m_id, max_m_id)]
            return p1_sim

    def get_p1_sim_sem(self, min_m_id, max_m_id, mTagTrueSem=None):
        """
        # p1:SBS-category-SBS 语义形式
        :param min_m_id:
        :param max_m_id:
        :param mTagTrueSem: tag的“语义形式”:使用各种特征提取方式 可以是'Deep','HDP','TF_IDF'等等
        :return:
        """
        if mTagTrueSem is not None:
            if self.p1_sims_sem is None:
                self.p1_sims_sem = self.load_sims('p1_sims_sem{}.dat'.format(mTagTrueSem))
            if (min_m_id, max_m_id) not in self.p1_sims_sem.keys():
                # print(min_m_id, max_m_id,'not in p1 sim!')
                self.flag1_sem = True
                p1_sim = cos_sim(self.mashup_tag_features[min_m_id], self.mashup_tag_features[max_m_id]) # 利用外部传入的tag feature,应该和参数名一致
                self.p1_sims_sem[(min_m_id, max_m_id)] = p1_sim
            else:
                p1_sim = self.p1_sims_sem[(min_m_id, max_m_id)]
            return p1_sim

    def get_p2_sim(self,min_m_id, max_m_id,mTextFalseSem=None):
        """
        # p1:SBS-text-SBS 非语义形式
        :param min_m_id:
        :param max_m_id:
        :param mTextFalseSem: 'EmbMax' p2,text的“非语义形式”
        :return:
        """
        # p2:SBS-content-SBS
        # 基于feature+cosine还是词汇embedding集合最大匹配
        if mTextFalseSem is not None:
            if self.p2_sims is None:
                self.p2_sims = self.load_sims('p2_sims{}.dat'.format(mTextFalseSem))

            if (min_m_id, max_m_id) not in self.p2_sims.keys ():
                # print('min_m_id, max_m_id,not in p2 sim!')
                self.flag2 = True
                # 1.编码过的文本索引也存储起来 2.不用每次编码，不需要加0，增加工作量...
                p2_sim = cpt_content_sim (self.ws, self.unpadded_encoded_mashup_texts[min_m_id], self.unpadded_encoded_mashup_texts[max_m_id])
                self.p2_sims[(min_m_id, max_m_id)] = p2_sim
            else:
                p2_sim = self.p2_sims[(min_m_id, max_m_id)]
            return p2_sim

    def get_p2_sim_sem(self, min_m_id, max_m_id, mTextTrueSem=None):
        # p2:SBS-content-SBS
        # 基于feature+cosine 使用各种特征提取方式 可以是'Deep','HDP','TF_IDF'等
        if mTextTrueSem is not None:
            if self.p2_sims_sem is None:
                self.p2_sims_sem = self.load_sims('p2_sims_sem{}.dat'.format(mTextTrueSem))

            if (min_m_id, max_m_id) not in self.p2_sims_sem.keys():
                # print('min_m_id, max_m_id,not in p2 sim!')
                self.flag2_sem = True
                # 利用外部传入的text feature,应该和参数名一致
                p2_sim = cos_sim(self.mashup_texts_features[min_m_id], self.mashup_texts_features[max_m_id])
                self.p2_sims_sem[(min_m_id, max_m_id)] = p2_sim
            else:
                p2_sim = self.p2_sims_sem[(min_m_id, max_m_id)]
            return p2_sim

    def get_p3_sim(self, m_id2, mashup_slt_apis_list):
        # p3:SBS-service-SBS
        # 待测样本使用已选择的service，历史使用全部
        if self.p3_sims is None:
            self.p3_sims = self.load_sims('p3_sims.dat')
        _key=(m_id2,tuple(mashup_slt_apis_list))
        if _key not in self.p3_sims.keys():
            self.flag3 = True
            p3_sim = cpt_p13_sim (mashup_slt_apis_list, self.mashup_apis[m_id2])
            self.p3_sims[_key] = p3_sim
        else:
            p3_sim=self.p3_sims[_key]
        return p3_sim

    def get_p4_sim(self, m_id2, mashup_slt_apis_list,aTagSem=None,api_tags = None):
        # p4:SBS-service-category-service-SBS
        # aTagSem: api的tag的模式  'MetaPath'或者其他形式 'Deep','HDP','TF_IDF'等
        if aTagSem is None:
            raise ValueError('must feed Para."aTagSem"!')

        if aTagSem == 'MetaPath':
            if self.p4_sims is None:
                self.p4_sims = self.load_sims('p4_sims_{}.dat'.format(aTagSem))
            self_sim_dict = self.p4_sims
        else: # 其他形式的tag特征
            if self.p4_sims_sem is None:
                self.p4_sims_sem = self.load_sims('p4_sims_sem{}.dat'.format(aTagSem))
            self_sim_dict = self.p4_sims_sem

        _key = (m_id2, tuple(mashup_slt_apis_list))
        if _key not in self_sim_dict.keys ():
            if aTagSem == 'MetaPath':
                self.flag4 = True
                if api_tags is None: # 用户不传入则使用默认
                    api_tags = self.unpadded_encoded_api_tags

                m1_api_category = {a_id: set (api_tags[a_id]) for a_id in mashup_slt_apis_list}
                m2_api_category = {a_id: set (api_tags[a_id]) for a_id in self.mashup_apis[m_id2]}
                p4_sim = cpt_p46_sim (m1_api_category, m2_api_category)
            else: # 使用传入的tag feature+最大化集合
                self.flag4_sem = True
                tag_sim_matrix = [[cos_sim(self.api_tag_features[a_id1],self.api_tag_features[a_id2])
                                   for a_id1 in mashup_slt_apis_list] for a_id2 in self.mashup_apis[m_id2]]
                p4_sim = cpt_2list_sim(tag_sim_matrix)
            self_sim_dict[_key] = p4_sim
        else:
            p4_sim = self_sim_dict[_key]
        return p4_sim

    def get_p5_sim(self, m_id2, mashup_slt_apis_list,aTextSem='EmbMax'):
        # p5: SBS-service-content-service-SBS  比较services的拼接文本相似度: IsRec使用的是最大化集合的做法!

        if self.p5_sims is None:
            self.p5_sims = self.load_sims('p5_sims{}.dat'.format(aTextSem))
        self_sim_dict = self.p5_sims

        _key=(m_id2,tuple(mashup_slt_apis_list))
        if _key not in self_sim_dict.keys():
            self.flag5 = True
            m2_apis = self.mashup_apis[m_id2]

            m1_api_texts,m2_api_texts=[],[]
            for api_id in mashup_slt_apis_list:
                m1_api_texts.extend(self.unpadded_encoded_api_texts[api_id]) # 未padding
            for api_id in m2_apis:
                m2_api_texts.extend(self.unpadded_encoded_api_texts[api_id])
            p5_sim = cpt_content_sim (self.ws, m1_api_texts, m2_api_texts)
            self_sim_dict[_key] = p5_sim
        else:
            p5_sim=self_sim_dict[_key]
        return p5_sim

    def get_p6_sim(self, m_id2, mashup_slt_apis_list):
        # p6:SBS-service-provider-service-SBS   !!!provider 信息还没处理！！！
        if self.p6_sims is None:
            self.p6_sims = self.load_sims('p6_sims.dat')
        _key = (m_id2, tuple(mashup_slt_apis_list))
        if _key not in self.p6_sims.keys ():
            self.flag6 = True
            # 存储起来...
            m1_api_provider = {a_id: set (self.api_id2provider[a_id]) for a_id in mashup_slt_apis_list}
            m2_api_provider = {a_id: set(self.api_id2provider[a_id]) for a_id in self.mashup_apis[m_id2]}
            p6_sim = cpt_p46_sim (m1_api_provider, m2_api_provider)
            self.p6_sims[_key] = p6_sim
        else:
            p6_sim = self.p6_sims[_key]
        return p6_sim

    def get_mashup_HIN_sims(self, m_id1, m_id2, mashup_slt_apis_list=None):
        # 有问题，待改！！！
        # 默认全部语义 基于feature
        """
        输入mashup的两个id，得到其根据HIN衡量的六种path的sim
        :param m_id1: 样本待测mashup
        :param m_id2: 历史mashup
        :param mashup_slt_apis_list  样本mashup已经选择的apis; 新场景下的部分冷启动需要输入
        :return: HIN衡量的六种path的sim的列表
        """
        if_mashup_sem,if_api_sem,if_mashup_sim_only,if_text_sem,if_tag_sem = new_Para.param.HIN_sim_paras
        # p1,p2只跟m_id1,m_id2有关,而p3 - p6跟m_id1, m_id2, mashup_slt_apis_list 有关
        min_m_id=min(m_id1, m_id2)
        max_m_id = max (m_id1, m_id2)

        p1_sim= self.get_p1_sim(min_m_id, max_m_id,if_mashup_sem,if_tag_sem)
        p2_sim= self.get_p2_sim(min_m_id, max_m_id,if_mashup_sem,if_text_sem)
        sim_list = [p1_sim, p2_sim] # if not self.newScence :

        if not if_mashup_sim_only: # 不是只使用mashup的1/2 path
            p3_sim = self.get_p3_sim(m_id2, mashup_slt_apis_list)
            p4_sim = self.get_p4_sim(m_id2, mashup_slt_apis_list,if_api_sem,if_tag_sem)
            p5_sim = self.get_p5_sim(m_id2, mashup_slt_apis_list,if_api_sem,if_text_sem)
            p6_sim = self.get_p6_sim(m_id2, mashup_slt_apis_list)
            sim_list.append(p3_sim)
            sim_list.append(p4_sim)
            sim_list.append(p5_sim)
            sim_list.append(p6_sim)

        return sim_list

    # 新的得到sim的方法
    def new_get_mashup_HIN_sims(self,m_id1, m_id2,mTagFalseSem=None,mashup_tags = None,mTagTrueSem=None,mTextFalseSem=None,mTextTrueSem=None):
        min_m_id=min(m_id1, m_id2)
        max_m_id = max (m_id1, m_id2)

        # print(mTagFalseSem,mashup_tags,mTagTrueSem,mTextFalseSem,mTextTrueSem)
        p1_sim= self.get_p1_sim(min_m_id, max_m_id,mTagFalseSem=mTagFalseSem,mashup_tags=mashup_tags)
        p1_sim_sem = self.get_p1_sim_sem(min_m_id, max_m_id,mTagTrueSem=mTagTrueSem)
        p2_sim= self.get_p2_sim(min_m_id, max_m_id,mTextFalseSem=mTextFalseSem)
        p2_sim_sem = self.get_p2_sim_sem(min_m_id, max_m_id, mTextTrueSem=mTextTrueSem)

        sim_list = [sim for sim in (p1_sim,p1_sim_sem,p2_sim,p2_sim_sem) if sim is not None]
        return sim_list

    def save_sim(self,flag, name, sim):
        sim_path = os.path.join(self.path, name)
        if flag:
            with open(sim_path, 'wb+') as f:
                pickle.dump(sim, f)  # 先存储
                print('save {},done!'.format(name))

    def save_changes(self):
        """
        新的调用之后，保存计算过的sim
        :return:
        """
        flags= [self.flag1,self.flag2,self.flag3,self.flag4,self.flag5,self.flag6]
        flags_sem = [self.flag1_sem, self.flag2_sem, self.flag4_sem]

        for i,sims in enumerate([self.p1_sims, self.p2_sims,self.p3_sims, self.p4_sims, self.p5_sims, self.p6_sims]):
            self.save_sim(flags[i],'p{}_sims.dat'.format(i+1), sims)

        sem_sims_list= [self.p1_sims_sem, self.p2_sims_sem, self.p4_sims_sem]
        sem_sims_index_list = [1,2,4]
        for i in range(len(sem_sims_list)):
            self.save_sim(flags_sem[i],'p{}_sims_sem{}.dat'.format(sem_sims_index_list[i],self.semantic_name), sem_sims_list[i])

    def save_changes(self,mTagFalseSem=None,mashup_tags = None,mTagTrueSem=None,mTextFalseSem=None,mTextTrueSem=None):
        """
        新的调用之后，保存计算过的sim
        :return:
        """
        if mTagFalseSem is not None :
            self.save_sim(self.flag1, 'p1_sims{}.dat'.format(mTagFalseSem),self.p1_sims)
        if mTagTrueSem is not None :
            self.save_sim(self.flag1_sem, 'p1_sims_sem{}.dat'.format(mTagTrueSem),self.p1_sims_sem)
        if mTextFalseSem is not None :
            self.save_sim(self.flag2, 'p2_sims{}.dat'.format(mTextFalseSem),self.p2_sims)
        if mTextTrueSem is not None :
            self.save_sim(self.flag2_sem, 'p2_sims_sem{}.dat'.format(mTextTrueSem),self.p2_sims_sem)
