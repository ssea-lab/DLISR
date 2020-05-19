import os
import pickle
import sys

from Helpers.util import cos_sim

sys.path.append ("..")

from embedding.encoding_padding_texts import get_default_encoding_texts
from main.processing_data import get_mashup_api_allCategories
import numpy as np
import random
from numpy.linalg import linalg
from numpy import mat

from Helpers.evaluator import evaluate
from main.evalute import summary
from main.new_para_setting import new_Para
from main.dataset import meta_data, dataset
from recommend_models.baseline import gensim_data, get_default_gd
# from sklearn.cluster import DBSCAN

class set_rec (object):
    def __init__(self, cluster_method,embedding_ts=pow(10,-5),cluster_ts=pow(10,-4), cluster_num=10,embedding_mode='W'):
        """
        :param cluster_ts: 聚类确定中心点的时候的阈值参数
        :param embedding_mode  使用哪种特征计算文本相似度
        """

        self.path = os.path.join (dataset.crt_ds.root_path, 'set_rec')  # 存放该模型的数据  0/set_rec
        # 存放中间结果是为了加速调参（聚类时的阈值参数）
        self.w_path = ''  # 学习W矩阵，计算文本相似度
        self.Uij_path = os.path.join (self.path, 'Uij.dat')  # 计算service之间的可组合型，内部索引，跟map结合才有意义
        self.all_sims_path = os.path.join (self.path, 'services_sims.dat')
        self.cluster_path= os.path.join (self.path, 'cluster.dat')
        self.embedding_mode=embedding_mode

        if not os.path.exists (self.path):
            os.makedirs (self.path)

        self.iter = 5
        self.init_lr = 0.01  # learning rate
        self.t_lr = 0.0
        self.reg = 0.01
        self.cluster_ts = cluster_ts
        self.embedding_ts=embedding_ts

        # 计算embedding,sim用
        self.m_features = []
        self.a_features = []

        self.D = None  # 字典大小
        self.K = 5  # latent factor in embedding

        if embedding_mode=='W':
            self.get_TF_IDF_features ('TF_IDF')
        else:
            self.get_TF_IDF_features(embedding_mode) # 'HDP'

        _, self.a_id2pop = meta_data.pd.get_api_co_vecs ()
        self.api2categories= self.get_services_1st_category(meta_data.pd) # 获取每个api的人工类别

        if embedding_mode=='W':
            self.A = np.random.randn (self.D, self.K)
            self.B = np.random.randn (self.D, self.K)
            self.C = np.identity (self.D)

            self.W = np.dot (self.A, self.B.T) + self.C  # D*D

        self.train_instances = []  # [(m_id,a_id1,a_id1),]
        self.get_train_instances ()

        # 计算u(si,sj)用
        self.m_id2local = None
        self.a_id2local = None
        self.D_e = None
        self.D_v = None
        self.H = None
        self.n_ij_dict = {}  # 基于内部索引

        self.deduct_cls = []
        self.api2score = {}
        self.api2sim = {}
        self.cls_num = cluster_num

        self.sorted_cls_index = None  # 对query最相近的cls的index排序
        self.cluster_method=cluster_method
        self.cls = []  # 存放每个类别的api id, [[],[],...]  聚类的结果
        self.core_ids = []
        self.aid2clsid={} # 知道每个service在哪个类中

    def get_services_1st_category(self,pd):
        """
        每个api的category是一个或多个类别的[]，也可能为空
        :param pd:
        :return:
        """
        api_id2info =pd.get_mashup_api_id2info ('api')
        id2category={}
        for api_id in range (len (self.a_id2pop)):
            categories=get_mashup_api_allCategories ('api', api_id2info, api_id, 'first')
            id2category[api_id]= categories[0] if len(categories)>0 else ''
        return id2category

    def get_TF_IDF_features(self,embedding_mode):
        """
        得到每个mashup和service的特征向量（tf-idf形式），字典形式：id：feature   真实id
        self.m_features={}
        self.a_features = {}
        :return:
        """
        # 未padding之前的文本
        gd = get_default_gd ()
        self.D = len (gd.dct)
        self.m_features, self.a_features = gd.model_pcs (embedding_mode,LDA_topic_num=20)
        np.save(os.path.join (self.path, 'm_features.dat'),self.m_features)
        np.save(os.path.join(self.path, 'a_features.dat'),self.a_features)
        print ('get_{}_features,done!'.format(embedding_mode))
        print ('m_features shape:{}'.format (self.m_features.shape))

    def get_train_instances(self):
        """
        构建特殊的样本实例：一正一负    真实id
        :return:
        """

        num_train_samples = len (dataset.crt_ds.train_mashup_id_list)
        train_m_ids = set ()
        inters = []  # 存储各个不同m-id的分界index
        for index in range (num_train_samples):
            temp_m_id = dataset.crt_ds.train_mashup_id_list[index]
            if temp_m_id not in train_m_ids:
                train_m_ids.add (temp_m_id)
                inters.append (index)
        inters.append (num_train_samples)  # 辅助最后一个

        for i in range (len (inters) - 1):
            start_index = inters[i]
            end_index = inters[i + 1]
            for index in range (end_index - start_index):
                if dataset.crt_ds.train_labels[index] == 1:
                    pos_index = index + start_index
                    neg_index = end_index - 1 - index
                    self.train_instances.append ((dataset.crt_ds.train_mashup_id_list[pos_index],
                                                  dataset.crt_ds.train_api_id_list[pos_index], dataset.crt_ds.train_api_id_list[neg_index]))
                else:
                    break

    def cpt_sim(self, fea1, fea2): # 'w','HDP','DL'
        """
        基于学到的W计算文本相似度：每个特征默认是行向量
        :param m_id:
        :param a_id:
        :return:
        """
        if self.embedding_mode=='W':
            return (fea1.dot (self.W)).dot (fea2)  # @注意！  一维向量.dot就是点乘；想得到二维乘积自己处理（先升维再dot太耗时
        else:
            return cos_sim(fea1,fea2) # 最一般的余弦相似度


    def update_embedding_paras(self, train=False, mode='mini_batch_grad',train_batch_size=128):
        """
        更新word embedding中A,B,C矩阵，方便计算相似度
        embedding_ts:两轮迭代的loss差
        :return:
        """
        if self.embedding_mode=='W':
            self.w_path = os.path.join (self.path, 'W_{}.dat.npy'.format (mode))  # 不同训练方式得到的W

            train_num = len (self.train_instances)
            print ('train_num:{}'.format (train_num))

            if not train and os.path.exists (self.w_path):
                self.W = np.load (self.w_path)
            else:
                last_loss = pow(10,10)
                for t in range (self.iter):
                    loss = 0.0
                    self.t_lr = self.init_lr / (1 + self.reg * self.init_lr * t)  # update the learning rate
                    temp1 = 1 - self.t_lr * self.reg
                    random.shuffle (self.train_instances)

                    if mode == 'mini_batch_grad':

                        num = train_num // train_batch_size
                        yushu = train_num % train_batch_size
                        if yushu != 0:
                            num += 1
                        for j in range (num):  # 每个mini batch
                            stop_index = train_num if (yushu != 0 and j == num - 1) else (j + 1) * train_batch_size
                            temp_batch_train_instances = self.train_instances[j * train_batch_size:stop_index]

                            temp_A_grad_part = 0  # 整个batch
                            temp_B_grad_part = 0
                            temp_C_grad_part = 0
                            Z_sum = 0

                            for m_id, a_id1, a_id2 in temp_batch_train_instances:  # batch内每个样本
                                # in the form of [,,]
                                m = self.m_features[m_id]
                                s1 = self.a_features[a_id1]
                                s2 = self.a_features[a_id2]
                                temp_sub = s1 - s2

                                #
                                Z = 1 - self.cpt_sim (m, s1) + self.cpt_sim (m, s2)
                                Z_sum += Z  # 整个batch关于Z的部分（怎么用？加负值还是0？

                                loss += (0 if Z <= 0 else Z + self.reg * 0.5 * linalg.norm (self.W))

                                temp_A_grad_part += np.expand_dims (m, axis=1).dot (np.expand_dims (temp_sub, axis=0))
                                temp_B_grad_part += np.expand_dims (temp_sub, axis=1).dot (np.expand_dims (m, axis=0))
                                temp_C_grad_part += np.dot (np.expand_dims (m, axis=1),
                                                            np.expand_dims (s1, axis=0)) - np.dot (
                                    np.expand_dims (m, axis=1), np.expand_dims (s2, axis=0))

                                """复杂度太高
                                temp_A_grad_part += simple_dot(m,temp_sub)
                                temp_B_grad_part += simple_dot (temp_sub,m)
                                temp_C_grad_part += simple_dot(m,s1) - simple_dot(m,s2)
                                """
                            # 整个batch更新一次
                            new_A = temp1 * self.A
                            new_B = temp1 * self.B
                            new_C = temp1 * self.C

                            if Z_sum > 0:
                                new_A = +self.t_lr * np.dot (temp_A_grad_part, self.B)
                                new_B = +self.t_lr * np.dot (temp_B_grad_part, self.A)
                                new_C = + self.t_lr * temp_C_grad_part

                            self.A = new_A
                            self.B = new_B
                            self.C = new_C

                            self.W = np.dot (self.A, self.B.T) + self.C
                            print ('iter {} ,bacth {},train done!'.format (t, j))

                    elif mode == 'STD':
                        for m_id, a_id1, a_id2 in self.train_instances:
                            new_A, new_B, new_C = None, None, None

                            # in the form of [,,]
                            m = self.m_features[m_id]
                            s1 = self.a_features[a_id1]
                            s2 = self.a_features[a_id2]
                            temp_sub = s1 - s2
                            #
                            Z = 1 - self.cpt_sim (m, s1) + self.cpt_sim (m, s2)
                            loss += (0 if Z <= 0 else Z + self.reg * 0.5 * linalg.norm (self.W))

                            new_A = temp1 * self.A
                            new_B = temp1 * self.B
                            new_C = temp1 * self.C

                            if Z > 0:
                                temp_A_grad_part += np.expand_dims (m, axis=1).dot (np.expand_dims (temp_sub, axis=0))
                                temp_B_grad_part += np.expand_dims (temp_sub, axis=1).dot (np.expand_dims (m, axis=0))
                                temp_C_grad_part += np.dot (np.expand_dims (m, axis=1),
                                                            np.expand_dims (s1, axis=0)) - np.dot (
                                    np.expand_dims (m, axis=1), np.expand_dims (s2, axis=0))
                                """
                                复杂度太高
                                new_A += simple_dot(m,temp_sub)
                                new_B += simple_dot (temp_sub,m)
                                new_C += simple_dot(m,s1) - simple_dot(m,s2)
                                """
                            self.A = new_A
                            self.B = new_B
                            self.C = new_C

                            self.W = np.dot (self.A, self.B.T) + self.C

                    print ("iter {}, loss:{}".format (t, loss))
                    if last_loss - loss < self.embedding_ts:
                        print ("early stop!")
                        break
                    last_loss = loss

                print ('update_embedding_paras and compute W,done!')
                np.save (self.w_path, self.W)
            print ('get W,done!')

    def transfer_2_local(self):
        m_ids, a_ids = zip (*dataset.crt_ds.train_mashup_api_list)
        m_ids = sorted (np.unique (m_ids))  # 按id从小到大排序
        a_ids = sorted (np.unique (a_ids))

        self.train_a_num=len(a_ids)
        self.train_m_num = len (m_ids)

        self.m_id2local = {m_ids[index]: index for index in range (self.train_m_num)}  # id到内部index的映射
        self.a_id2local = {a_ids[index]: index for index in range (self.train_a_num)}

        m_times_list = np.zeros ((self.train_m_num))  # 度数，调用次数，包含api数
        a_times_list = np.zeros ((self.train_a_num))
        self.H = np.zeros ((self.train_a_num,self.train_m_num ))
        for m_id, a_id in dataset.crt_ds.train_mashup_api_list:
            m_times_list[self.m_id2local[m_id]] += 1
            a_times_list[self.a_id2local[a_id]] += 1
            self.H[self.a_id2local[a_id],self.m_id2local[m_id] ] = 1

        self.D_e = np.diag (m_times_list)
        self.D_v = np.diag (a_times_list)

    def cpt_u_ij(self):
        """
        计算一对service的可组合性
        :return:
        """
        if os.path.exists (self.Uij_path):
            self.n_ij_dict = pickle.load (open(self.Uij_path, 'rb'))
        else:
            L = self.D_v - (self.H.dot (np.array (mat (self.D_e).I))).dot (self.H.T)
            L_plus = cpt_Moore_Penrose (L)

            V_G = 0
            for i in range (self.train_a_num):
                V_G += self.D_v[i][i]

            I = np.identity (self.train_a_num)

            max_n_ij = -1
            # 计算任意两个service之间的n
            for i in range (self.train_a_num):  # 大到小！ 内部索引！！！
                for j in range (i):
                    I_sub = I[i] - I[j]
                    n_ij = V_G * (I_sub.dot (L_plus)).dot (I_sub)  # 最终得到一个值
                    self.n_ij_dict[(i, j)] = n_ij

                    if n_ij > max_n_ij:
                        max_n_ij = n_ij

            for pair, value in self.n_ij_dict.items ():  # 归一化
                self.n_ij_dict[pair] = (1 - value) / max_n_ij

            print ('cpt_u_ij,done!')
            pickle.dump (self.n_ij_dict, open(self.Uij_path,'wb'))

    def get_US_ij(self, a_id1, a_id2):  # 要转化为内部index！！！
        if len (self.n_ij_dict) == 0:
            self.transfer_2_local ()  # 转化为局部索引
            self.cpt_u_ij ()
        local_a_id1=self.a_id2local[a_id1]
        local_a_id2 = self.a_id2local[a_id2]
        # 内部索引，大到小
        return self.n_ij_dict[(local_a_id1, local_a_id2)] if local_a_id1> local_a_id2 else self.n_ij_dict[(local_a_id2, local_a_id1)]

    def clustering(self,max_iter=100):
        """全程是对service id的操作，涉及到获取相似度和pop"""
        print ('clustering......!')
        # a_ids = [i for i in range (len (self.a_id2pop))] 有些api在train集中未出现过！！！不能用？？？
        m_ids, a_ids = zip (*dataset.crt_ds.train_mashup_api_list)
        a_ids = sorted (np.unique (a_ids))  # 按id从小到大排序

        sorted_a_ids = sorted (a_ids, key=lambda a_id: self.a_id2pop[a_id], reverse=True)  # 记住写法！！

        if self.cluster_method[-6:]=='kmeans':
            # 初始化
            temp_a_id = sorted_a_ids[0]
            self.cls.append ([temp_a_id])
            self.aid2clsid[temp_a_id]=len(self.cls)-1
            self.core_ids.append (temp_a_id)

            # 确定K个core:
            if self.cluster_method=='original_kmeans':
                all_sims = []
                # single——paas
                for a_id in sorted_a_ids[1:]:
                    flag = False
                    for cls_index in range (len (self.cls)):
                        sim = self.cpt_sim (self.a_features[a_id], self.a_features[self.core_ids[cls_index]])  # 到某个core的sim
                        all_sims.append (sim)
                        if sim <= self.cluster_ts:
                            flag = True
                            break
                    if not flag:  # 不属于任何一个已有core
                        self.cls.append ([a_id])
                        self.aid2clsid[a_id] = len(self.cls) - 1
                        self.core_ids.append (a_id)

                    if len (self.cls) >= self.cls_num:
                        break
                np.save (self.all_sims_path, np.array (all_sims))  # 存储每个service之间sim，查看，简单判断阈值设为多少合适

            # # 借助人工标注的类标签和pop，选取core
            # elif self.cluster_method=='manner_kmeans':
            #     core_categories=set(self.api2categories[temp_a_id]) #core对应的类别名
            #     for a_id in sorted_a_ids[1:]:
            #         temp_category=self.api2categories[a_id]
            #         if temp_category !='' and temp_category not in core_categories:
            #             self.cls.append ([a_id])
            #             self.core_ids.append (a_id)
            #             core_categories.add(temp_category)
            #
            #         if len (self.cls) >= self.cls_num:
            #             break
            print ('select core services,done!')
            np.save (self.cluster_path, np.array (self.cls))  #

            self.cls_num=len(self.cls) # 阈值设置的太大时，可能类别数目没有10个

            # 第一次先根据到core的距离初步划分
            for a_id in sorted_a_ids:
                if a_id not in self.core_ids:
                    max_sim = -1
                    cls_index = -1
                    for temp_cls_index in range (self.cls_num):
                        temp_sim=self.cpt_sim (self.a_features[a_id], self.a_features[self.core_ids[temp_cls_index]])
                        if temp_sim > max_sim:
                            cls_index = temp_cls_index
                            max_sim = temp_sim
                    self.cls[cls_index].append(a_id)
                    self.aid2clsid[a_id] = cls_index

            # 对非core api进行K-means聚类，保证每个core一直不变
            iter_num=0
            while iter_num<=max_iter:
                change_times = 0
                for a_id in sorted_a_ids:
                    if a_id not in self.core_ids:
                        max_sim = -1
                        new_cls_index = -1
                        old_cls_index = self.aid2clsid[a_id]
                        for cls_index in range (self.cls_num):
                            temp_center_fea = np.average (
                                np.array ([self.a_features[temp_a_id] for temp_a_id in self.cls[cls_index]]), axis=0)
                            temp_sim = self.cpt_sim (self.a_features[a_id], temp_center_fea)
                            if temp_sim > max_sim:
                                new_cls_index = cls_index
                                max_sim = temp_sim
                        if old_cls_index != new_cls_index:
                            self.cls[old_cls_index].remove (a_id)
                            self.cls[new_cls_index].append (a_id)
                            self.aid2clsid[a_id]=new_cls_index
                            change_times += 1

                if change_times == 0:  # 本轮迭代无变动
                    break
                iter_num+=1

        elif self.cluster_method=='manner':
            # 根据人工category分类
            # 没有类别标签的化为一类 others
            core_categories = ['others']  # core对应的类别名
            self.cls.append ([])

            for a_id in sorted_a_ids:
                temp_category = self.api2categories[a_id]
                if temp_category == '': # others
                    self.cls[0].append (a_id)
                else:
                    if temp_category not in core_categories:
                        self.cls.append ([a_id])
                        core_categories.append (temp_category)
                    else:
                        cls_index= core_categories.index(temp_category)
                        self.cls[cls_index].append (a_id)
            if len(self.cls[0])==0:
                del(self.cls[0])
            print ('name of category:')
            print(core_categories)
        self.cls_num = len (self.cls)

        print('num of all categories:')
        lens=[len(cls) for cls in self.cls]
        print(len(lens))

        print ('clustering,done!')

    def cls_deduction(self, query_feature, a_wight=0.6, b_wight=0.15, topK=5):
        """
        针对每个query进行一次;同时完成cls对query的排序，和query内部的裁剪；每个api对mashup的sim可重用
        """
        self.deduct_cls = []  #
        self.api2sim = {}  # 每个api与query的sim
        self.api2score = {}  # 每个api针对某个query的文本相似度和qos值的效度，可重用

        cls_index_2_average_sim = {}
        for cls_index in range (self.cls_num):
            cls = self.cls[cls_index]
            sim_sum = 0
            for a_id in cls:
                sim = self.cpt_sim (self.a_features[a_id], query_feature)
                sim_sum += sim
                self.api2sim[a_id] = sim
                self.api2score[a_id] = a_wight * sim + b_wight * self.a_id2pop[a_id]
            cls_index_2_average_sim[cls_index] = sim_sum / len (cls)  # 先进行求平均值？

            self.deduct_cls.append (sorted (cls, key=lambda a_id: self.api2score[a_id], reverse=True)[:min(topK,len(cls))])
        sorted_cls_index = [i for i in range (self.cls_num)]
        self.sorted_cls_index = sorted (sorted_cls_index, key=lambda cls_index: cls_index_2_average_sim[cls_index],
                                        reverse=True)
        # print ('cls_deduction,done!')

    def score_set(self, result_set):
        """
        某个推荐结果（api id的list）的总效度函数
        :param result_set:
        :return:
        """
        utity_sum = 0
        for i in range (len (result_set)):
            for j in range (i):
                utity_sum += self.get_US_ij (result_set[i], result_set[j])
        for api_id in result_set:
            utity_sum += self.api2score[api_id]
        return utity_sum

    def recommend(self, topKs=[5]):
        """
        外部调用的推荐方法
        :param cluster_threshold:
        :param topKs: NDCG@5 / 10
        :return:
        """
        self.update_embedding_paras ()  # 先更新embedding 参数，可以计算sim
        self.clustering ()  # 10个类

        all_indicators=[] # 第一维是mashup；第二维是top5/10；第三维是每个指标

        test_m_num = len (dataset.crt_ds.test_mashup_id_list)
        for i in range (test_m_num):
            a_mashup_indicators=[]

            test_m_id = dataset.crt_ds.test_mashup_id_list[i][0]  # 每个mashup id
            query_feature = self.m_features[test_m_id]  # 对每个测试的mashup/query

            self.cls_deduction (query_feature, a_wight=0.6, b_wight=0.15, topK=5)  # 裁剪和挑选最近似cls  每个类选择几个

            for topK in topKs:  # NDCG@5 or NDCG@10
                candidate_results = combinations (
                    [self.deduct_cls[cls_index] for cls_index in self.sorted_cls_index[:topK]]).get_results () # 选择topK个类进行组合
                index2utity = {}
                for index in range (len (candidate_results)):
                    index2utity[index] = self.score_set (candidate_results[index])
                sorted_index = sorted ([i for i in range (len (candidate_results))],
                                       key=lambda index: index2utity[index], reverse=True)  # 最终推荐的
                max_k_candidates = candidate_results[sorted_index[0]]  # 这里只使用一个set？
                # print(max_k_candidates)

                # 调用evalute函数
                a_mashup_indicators.append(list (evaluate (max_k_candidates, dataset.crt_ds.grounds[i], topK)))  # 评价得到五个指标，K对NDCG等有用

            all_indicators.append(a_mashup_indicators)
        all_indicators=np.average(all_indicators,axis=0)
        summary (new_Para.param.evaluate_path,
                 'set_rec_{}_clustingTs:{}_clsNum:{}'.format (self.cluster_method, self.cluster_ts, self.cls_num),
                 all_indicators, topKs)  # 名字中再加区别模型的参数


def cpt_Moore_Penrose(L):
    U, S, V = linalg.svd (L)  # 求出的V实际是（共轭转置，实数相当于直接转置）转置后的结果！
    num = len (S)  # 非零奇异值的个数,肯定不大于min_num
    num_U = len (U)
    num_V = len (V)
    min_num = min (num_U, num_V)

    S_matrix = np.zeros ((num_U, num_V))  # 构造二维S矩阵
    S_plus = np.zeros ((num_U, num_V))
    for i in range (num):
        S_matrix[i][i] = S[i]
        S_plus[i][i] = 1.0 / S[i]

    L_plus = np.dot (V.T, S_plus.T).dot (U.T)  # weki
    print ('cpt_Moore_Penrose to compute u_ij,done!')
    return L_plus


class combinations (object):
    """
    对所有的categories，每个选择一个api，进行组合
    """

    def __init__(self, lists):
        self.lists = lists
        self.final_len = len (lists)
        self.results = []

    def get_combinations(self, a_list):
        if len (a_list) == self.final_len:
            self.results.append (a_list)
            return

        for a_id in self.lists[len (a_list)]:
            new_a_list = list (a_list)
            new_a_list.append (a_id)
            self.get_combinations (new_a_list)

    def get_results(self):
        self.get_combinations ([])
        return self.results


def simple_dot(col, row):
    """
    一个列向量乘以行向量(但是二者这里都是一维行向量的形式)，得到一个D*D矩阵
    直接拓展成2D，再dot貌似效率较低（虽然很多0，但是也要计算一次？）D^3
    这里加速计算
    :return:
    """
    num1 = len (col)
    num2 = len (row)
    result = np.zeros ((num1, num2))
    for i in range (num1):
        for j in range (num2):
            result[i, j] = col[i] * row[j]
    return result


if __name__ == '__main__':
    cluster_tss=[pow (10,-3),pow (10,-4),pow (10,-5)]
    cluster_nums=[10,30,50]
    for cluster_ts in cluster_tss:
        for cluster_num in cluster_nums:
            set_rec ('manner',embedding_ts=pow(10,-5), cluster_ts=pow (10, -4), cluster_num=10).recommend ([5])  # 'original_kmeans','manner_kmeans','manner'
            set_rec ('original_kmeans', embedding_ts=pow (10, -5), cluster_ts=cluster_ts, cluster_num=cluster_num).recommend ([5])