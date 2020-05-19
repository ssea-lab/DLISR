import os
import random
import sys
sys.path.append("..")

from Helpers.util import read_2D_list, save_2D_list
from main.new_para_setting import new_Para
from .processing_data import process_data
from mf.get_UI import get_UV
from Helpers.util import read_split_train, read_test_instance, save_split_train, save_test_instance
import numpy as np

# 根据生成的训练测试集，生成一个dataset对象，里面实现了各种方法，更方便后续处理(train_data,test_data,his_mashup_ids的各种数据)
# 访问时，使用dataset.crt_ds 访问当前使用的对象
# dataset.UV_obj 访问根据当前训练集生成的get_UV()对象，跟MF相关的数据

class dataset(object): # 已划分的数据集
    crt_ds = None
    UV_obj = None

    @classmethod
    def set_current_dataset(cls,dataset): # 该类当前使用的dataset数据对象,必须先设置
        cls.crt_ds = dataset
        cls.UV_obj = dataset.get_UV_obj()

    def __init__(self,root_path,name,kcv_index=0):
        self.kcv_index = kcv_index
        # 数据名 eg: newScene_neg_{}_sltNum{}_com_{}_trainPos_{}_testCandi{}_kcv_{}
        self.data_name = new_Para.param.data_mode+'_'+name + '_kcv{}_'.format(kcv_index)
        # self.no_kcv_root_path = root_path # 不区分kcv的路径
        self.root_path = os.path.join (root_path , str(kcv_index)) # 存放数据的根路径
        self.train_instances_path = os.path.join(self.root_path, 'train_instances_no_slt.dat')
        self.test_instances_path = os.path.join(self.root_path, 'test_instances_no_slt.dat')
        self.all_ground_api_ids_path = os.path.join(self.root_path, 'all_ground_api_ids.dat')
        # 新场景多存储slt_ids
        if new_Para.param.data_mode == 'newScene':
            self.train_slt_ids_path = os.path.join(self.root_path, 'train_slt_ids.dat')
            self.test_slt_ids_path = os.path.join(self.root_path, 'test_slt_ids.dat')

        self.train_data,self.test_data = None,None
        self.UV_obj = None

    def set_data(self,train_mashup_id_instances, train_api_id_instances,train_labels,train_slt_ids, test_mashup_ids,all_candidate_api_ids,all_ground_api_ids,test_slt_ids):
        # 新场景下的数据，划分后首次设置
        self.train_mashup_id_list= train_mashup_id_instances
        self.train_api_id_list = train_api_id_instances
        self.train_labels = train_labels
        self.slt_api_ids_instances = train_slt_ids
        self.test_mashup_id_list = test_mashup_ids
        self.test_api_id_list = all_candidate_api_ids
        self.grounds = all_ground_api_ids
        self.test_slt_ids = test_slt_ids
        self.train_mashup_api_list = list(zip(self.train_mashup_id_list,self.train_api_id_list))

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        self.save() # 首次划分后存储
        self.set_others() # 设置路径等
        self.set_train_test_data() # 设置train test
        self.save_true_train_data() # 训练集中的正例，供libRec等使用

    def set_data_oldScene(self,train_mashup_id_instances, train_api_id_instances,train_labels, test_mashup_ids,all_candidate_api_ids,all_ground_api_ids):
        # 旧场景下的数据
        self.train_mashup_id_list= train_mashup_id_instances
        self.train_api_id_list = train_api_id_instances
        self.train_labels = train_labels
        self.test_mashup_id_list = test_mashup_ids
        self.test_api_id_list = all_candidate_api_ids
        self.grounds = all_ground_api_ids
        self.train_mashup_api_list = list(zip(self.train_mashup_id_list,self.train_api_id_list))

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        self.save() # 第一次划分dataset时，要存储训练测试集；以后直接read_data()读取即可
        self.set_others() # 设置其他，eg：模型路径等
        self.set_train_test_data() # 设置train test

    def save(self):
        # 存储训练测试样本集
        save_split_train(self.train_instances_path, self.train_mashup_api_list, self.train_labels)
        save_test_instance(self.test_instances_path, self.test_mashup_id_list, self.test_api_id_list)
        save_2D_list(self.all_ground_api_ids_path, self.grounds)
        # 新场景多存储slt_ids
        if new_Para.param.data_mode == 'newScene':
            save_2D_list(self.train_slt_ids_path, self.slt_api_ids_instances)
            save_2D_list(self.test_slt_ids_path, self.test_slt_ids)

    def read_data(self):
        # 读取已有的训练测试集
        self.train_mashup_api_list, self.train_labels = read_split_train(self.train_instances_path, True)
        print('data train samples num:{}'.format(len(self.train_labels)))
        self.train_mashup_id_list, self.train_api_id_list = zip(*self.train_mashup_api_list)
        self.test_mashup_id_list, self.test_api_id_list = read_test_instance(self.test_instances_path)
        self.grounds = read_2D_list(self.all_ground_api_ids_path)
        if new_Para.param.data_mode == 'newScene':
            self.slt_api_ids_instances = read_2D_list(self.train_slt_ids_path)
            self.test_slt_ids = read_2D_list(self.test_slt_ids_path)

        self.set_others()
        self.set_train_test_data()  # 设置train test
        self.save_true_train_data()

    # 好像暂时没用??? 直接用了get_UV_obj()
    def save_true_train_data(self):
        # 存储训练集中的正例，供lirbec使用
        # 但是选择的服务可能不同，所以同一个m_id,a_id对可能出现多次？？？ 所以新场景下的MF不用这个数据处理方法？
        true_train_set_path = os.path.join(self.root_path, 'train_set.data')
        if not os.path.exists(true_train_set_path):
            true_train_mashup_api_pairs =[]
            for index,label in enumerate(self.train_labels):
                if label:
                    true_train_mashup_api_pairs.append(self.train_mashup_api_list[index])
            save_2D_list(true_train_set_path,true_train_mashup_api_pairs)
            return true_train_mashup_api_pairs

    def set_others(self):
        # 在set_data()或read_data后设置
        self.his_mashup_ids = np.unique(self.train_mashup_id_list)  # 训练mashup id的有序排列
        print('data train mashup_id num:{}'.format(len(self.his_mashup_ids)))

        # 模型随数据变化，所以存储在数据的文件夹下
        self.model_path = os.path.join(self.root_path, '{}')  # simple_model_name  CI路径
        self.new_best_epoch_path = os.path.join('{}', 'best_epoch.dat')  # model_dir,  .format(simple_model_name)
        self.new_model_para_path = os.path.join('{}', 'weights_{}.h5')   # model_dir, .format(simple_model_name, epoch)
        self.new_best_NDCG_path = os.path.join('{}', 'best_NDCG.dat')  # model_dir,  .format(simple_model_name)

    def get_UV_obj(self): # 'pmf','BPR','listRank','Node2vec'
        # 根据train mashup_ids 将全部数据集划分，得到训练数据集，供MF使用
        if self.UV_obj is None:
            true_train_mashup_api_list = [(m_id,a_id)  for m_id,a_id in meta_data.mashup_api_list if m_id in self.his_mashup_ids]
            # UV对象中mashup id等也是按顺序排列，跟his_mashup_ids 一样
            self.UV_obj= get_UV (self.root_path, new_Para.param.mf_mode, true_train_mashup_api_list)
        return self.UV_obj

    def set_train_test_data(self):
        # 设置训练和测试数据
        if self.train_data is None:
            if not new_Para.param.pairwise: #
                # 只有新场景下且需要slt apis时
                if new_Para.param.data_mode=='newScene' and new_Para.param.need_slt_apis:
                    self.train_data = self.train_mashup_id_list, self.train_api_id_list, self.slt_api_ids_instances, self.train_labels
                    self.test_data = self.test_mashup_id_list, self.test_api_id_list, self.test_slt_ids, self.grounds
                else:
                    self.train_data = self.train_mashup_id_list, self.train_api_id_list, self.train_labels
                    self.test_data = self.test_mashup_id_list, self.test_api_id_list, self.grounds
            else: # pairwise型的训练数据，根据pointwise型的转化
                dict_pos = {}
                dict_neg = {}
                for index in range(len(self.train_mashup_id_list)):
                    mashup_id=self.train_mashup_id_list[index]
                    api_id=self.train_api_id_list[index]
                    slt_api_ids=self.slt_api_ids_instances[index]

                    key = (mashup_id, tuple(slt_api_ids)) if new_Para.param.data_mode=='newScene' and new_Para.param.need_slt_apis else mashup_id

                    if key not in dict_pos.keys():
                        dict_pos[key]=[]
                    if key not in dict_neg.keys():
                        dict_neg[key] = []
                    if self.train_labels[index]==1:
                        dict_pos[key].append(api_id) # 可以包含多个正例
                    else:
                        dict_neg[key].append(api_id)

                train_mashup_id_list, train_pos_api_id_list, slt_api_ids_instances,train_neg_api_id_list=[],[],[],[]
                for key in dict_pos.keys():
                    # assert len()
                    pos_api_ids=dict_pos[key]*new_Para.param.num_negatives
                    train_pos_api_id_list.extend(pos_api_ids)
                    neg_api_ids=dict_neg[key]
                    train_neg_api_id_list.extend(neg_api_ids)
                    pair_num =len(neg_api_ids)
                    train_mashup_ids = [key[0]]*pair_num
                    train_mashup_id_list.extend(train_mashup_ids)
                    if new_Para.param.data_mode=='newScene' and new_Para.param.need_slt_apis:
                        slt_api_ids= list(key[1])
                        for i in range(pair_num):
                            slt_api_ids_instances.append(slt_api_ids)
                train_labels = [1]*len(train_mashup_id_list) # 随便设，占位而已

                if new_Para.param.data_mode=='newScene' and new_Para.param.need_slt_apis:
                    self.train_data = train_mashup_id_list, train_pos_api_id_list, slt_api_ids_instances,train_neg_api_id_list, train_labels
                    self.test_data = self.test_mashup_id_list, self.test_api_id_list, self.test_slt_ids, self.grounds
                else:
                    self.train_data = train_mashup_id_list, train_pos_api_id_list,train_neg_api_id_list, train_labels
                    self.test_data = self.test_mashup_id_list, self.test_api_id_list, self.grounds

        return self.train_data,self.test_data

    # 下面的几个方法是转化训练，测试集，方便测试其他模型
    def reduct(self,data):
        Mid_Aid_lables = {}
        _zip = zip(data[0],data[1])
        for index,key in enumerate(_zip):
            if tuple(key) not in set(Mid_Aid_lables.keys()):
                Mid_Aid_lables[key]=data[-1][index]
        _1,_2 = zip(*Mid_Aid_lables.keys())
        return (list(_1),list(_2),list(Mid_Aid_lables.values()))

    def transfer(self):
        # 将无slt apis的含重复数据去重   'newScene'且need_slt_apis=False时
        if self.train_data is None:
            self.set_train_test_data()
        print('before reduction, train samples:{}'.format(len(self.train_data[0])))
        self.train_data= self.reduct(self.train_data)
        print('after reduction, train samples:{}'.format(len(self.train_data[0])))

        self.test_data_no_reduct = self.test_data # 只评价去重后的，样本太少，不太公平
        test_mashup_ids_set=set()
        test_mashup_id_list, test_api_id_list, grounds=[],[],[]
        for index,test_mashup_ids in enumerate(self.test_mashup_id_list):
            if test_mashup_ids[0] not in test_mashup_ids_set:
                test_mashup_ids_set.add(test_mashup_ids[0])
                test_mashup_id_list.append(test_mashup_ids)
                test_api_id_list.append(self.test_api_id_list[index])
                grounds.append(self.grounds[index])
        print('before reduction, test samples:{}'.format(len(self.test_data[0])))
        self.test_data = test_mashup_id_list, test_api_id_list, grounds
        print('after reduction, test samples:{}'.format(len(self.test_data[0])))
        print('remove_,done!')

    def transfer_false_test_DHSR(self,if_reduct_train= False):
        # 为了新场景下让DHSR/SVD等可以work，假设它们可以根据用户选择实时训练，把测试集中已选择的服务，加入训练集
        # 使用跟我们的模型类似的self.train_data,self.test_data
        # 把测试集分为几种情况： 选择一个服务的(只有一个作为正例)；2个的；3个的。分别跟训练集整合在一起，作为综合训练集

        # 训练集不做改变，不同已选，相同的正例，多次出现也无所谓，跟MISR的数据保持一致

        if if_reduct_train: # 约减原始训练集中的大量重复信息，否则显得测试集的伪训练集很小
            self.train_mashup_id_list,self.train_api_id_list, self.train_labels = self.reduct(self.train_data)

        self.train_data,self.test_data = [],[] # 改变格式，按照测试集已选的数目，生成几个不同的训练和测试
        all_apis = {api_id for api_id in range(meta_data.api_num)}
        num_negatives = new_Para.param.num_negatives

        # 把测试集的数据转化为训练集，按照已选服务个数划分
        # 分别训练测试，得到1,2,3场景下的指标
        def certain_slt_num_split(slt_num):
            set_ = set() # 存储某个mashup，某个长度已选的数据的集合
            train_mashup_id_list, train_api_id_list, labels = list(self.train_mashup_id_list),list(self.train_api_id_list),list(self.train_labels)
            test_mashup_id_list, test_api_id_list, grounds=[],[],[]
            for index,test_mashup_ids in enumerate(self.test_mashup_id_list):
                m_id, slt_api_ids = test_mashup_ids[0],self.test_slt_ids[index]
                if len(slt_api_ids) == slt_num and (m_id,len(slt_api_ids)) not in set_:
                    train_mashup_id_list.extend([m_id]*slt_num*(num_negatives+1)) # 测试集中已选的服务作为正例,还有负例

                    train_api_id_list.extend(slt_api_ids)
                    neg_api_ids = list(all_apis-set(slt_api_ids))
                    random.shuffle(neg_api_ids)
                    neg_api_ids = neg_api_ids[:slt_num * num_negatives]
                    train_api_id_list.extend(neg_api_ids)

                    labels.extend([1] * slt_num)
                    labels.extend([0] * slt_num* num_negatives)

                    # 同时也需要测试，跟原来格式相同
                    test_mashup_id_list.append(test_mashup_ids)
                    test_api_id_list.append(self.test_api_id_list[index])
                    grounds.append(self.grounds[index])

            self.train_data.append((train_mashup_id_list, train_api_id_list, labels))
            self.test_data.append((test_mashup_id_list, test_api_id_list, grounds))

        for i in range(1,new_Para.param.slt_item_num+1):
            print('slt_num:',i)
            print('before, train samples:{}'.format(len(self.train_mashup_id_list)))
            certain_slt_num_split(i)
            print('after, train samples:{},{}'.format(len(self.train_data[-1][0]),len(self.train_data[-1][-1])))
            print('after, test samples:{}'.format(len(self.test_data[i-1][0])))
        print('transfer for DHSR,done!')
        return self.train_data,self.test_data

    def transfer_false_test_MF(self):
        # 只需要返回一个train_mashup_api_list
        # 把测试集的数据转化为训练集，按照已选服务个数划分
        # 分别训练测试，得到1,2,3场景下的指标

        # 正例训练集
        train_mashup_id_list, train_api_id_list = [], []
        Mid_Aid_set = set()
        _zip = zip(self.train_data[0], self.train_data[1])
        train_labels = self.train_data[-1]
        for index, Mid_Aid_pair in enumerate(_zip):
            if train_labels[index] and tuple(Mid_Aid_pair) not in Mid_Aid_set:  # 正例且之前未出现过
                train_mashup_id_list.append(Mid_Aid_pair[0])
                train_api_id_list.append(Mid_Aid_pair[1])

        def certain_slt_num_split(train_mashup_id_list,train_api_id_list,slt_num):
            # 测试集
            set_ = set() # 存储某个mashup，某个长度已选的数据的集合
            test_mashup_id_list, test_api_id_list, grounds=[],[],[]
            for index,test_mashup_ids in enumerate(self.test_mashup_id_list):
                m_id, slt_api_ids = test_mashup_ids[0],self.test_slt_ids[index]
                if len(slt_api_ids) == slt_num and (m_id,len(slt_api_ids)) not in set_:
                    train_mashup_id_list.extend([m_id]*slt_num) # 测试集中已选的服务作为正例,还有负例
                    train_api_id_list.extend(slt_api_ids)

                    # 同时也需要测试，跟原来格式相同
                    test_mashup_id_list.append(test_mashup_ids)
                    test_api_id_list.append(self.test_api_id_list[index])
                    grounds.append(self.grounds[index])

            self.train_data.append(list(zip(train_mashup_id_list, train_api_id_list))) # 供get_U_V使用
            self.test_data.append((test_mashup_id_list, test_api_id_list, grounds))

        self.train_data,self.test_data = [],[] # 改变格式，按照测试集已选的数目，生成几个不同的训练和测试
        for i in range(1,new_Para.param.slt_item_num+1):
            print('slt_num:',i)
            print('before, train samples:{}'.format(len(self.train_mashup_id_list)))
            certain_slt_num_split(list(train_mashup_id_list), list(train_api_id_list), i)
            print('after, train samples:{}'.format(len(self.train_data[-1])))
            true_train_set_path = os.path.join(self.root_path, 'train_set_MF_{}.data'.format(i))
            save_2D_list(true_train_set_path, self.train_data[i-1]) # 把训练集(加上了某个长度的测试集)存起来，java处理
        print('transfer for MF,done!')
        return self.train_data, self.test_data

    # 好像没用到，直接在全部测试后再划分就可以了，见evaluate

    # def split_test_ins_by_sltNum(self):
    #     # 将测试集按照已选api的数目分类,分别测试和评价,返回的是三种不同长度slt_api_ids的样本列表：格式跟train_data相同
    #     # 四种数据:mashup_id,api_id,test_slt_ids,grounds
    #     # 或三种数据:mashup_id,api_id,grounds
    #     num = 4 if new_Para.param.need_slt_apis else 3
    #     l1 = [[] for i in range(num)]  # [[], [], [], []]
    #     l2 = [[] for i in range(num)]
    #     l3 = [[] for i in range(num)]
    #
    #     # print(len(self.test_mashup_ids),len(self.all_candidate_api_ids),len(self.all_ground_api_ids),len(self.test_slt_ids))
    #
    #     for index in range(len(self.test_mashup_id_list)):
    #         _len = len(self.test_slt_ids[index])
    #         if _len == 1:
    #             l = l1
    #         elif _len == 2:
    #             l = l2
    #         elif _len == 3:
    #             l = l3
    #         if new_Para.param.need_slt_apis:
    #             l[0].append(self.test_mashup_id_list[index])
    #             l[1].append(self.test_api_id_list[index])
    #             l[2].append(self.test_slt_ids[index])
    #             l[3].append(self.grounds[index])
    #         else:
    #             l[0].append(self.test_mashup_id_list[index])
    #             l[1].append(self.test_api_id_list[index])
    #             l[2].append(self.grounds[index])
    #     return l1, l2, l3

    def get_few_samples(self, train_num, test_num=32):
        """
        测试模型时只利用一小部分样本做训练和测试
        :param train_num:
        :return:
        """
        if self.train_data is None:
            self.set_train_test_data()
        if new_Para.param.data_mode=='newScene' and new_Para.param.need_slt_apis:
            return ((self.train_data[0][:train_num], self.train_data[1][:train_num], self.train_data[2][:train_num], self.train_data[3][:train_num]),
                    (self.test_data[0][:test_num], self.test_data[1][:test_num], self.test_data[2][:test_num], self.test_data[3][:test_num]))
        else:
            return ((self.train_data[0][:train_num], self.train_data[1][:train_num], self.train_data[2][:train_num]),
                    (self.test_data[0][:test_num], self.test_data[1][:test_num], self.test_data[2][:test_num]))

class meta_data(object):
    # 获取pd对象，可利用内容信息和调用关系数据
    # 未划分
    @classmethod
    def initilize(cls):
        cls.data_dir = new_Para.param.data_dir
        cls.pd = process_data(cls.data_dir, False)
        cls.mashup_id2info = cls.pd.get_mashup_api_id2info('mashup')
        cls.api_id2info = cls.pd.get_mashup_api_id2info('api')

        cls.mashup_api_list = cls.pd.get_mashup_api_pair('list')  # 未划分的数据集, UI矩阵的list形式
        print('num of pairs:{}'.format(len(cls.mashup_api_list)))
        cls.mashup_num = cls.pd.mashup_num
        cls.api_num = cls.pd.api_num
        print('num of mashup or api:{},{}'.format(cls.mashup_num, cls.api_num))

        cls.mashup_descriptions, cls.api_descriptions, cls.mashup_categories, cls.api_categories = cls.pd.get_all_texts(new_Para.param.Category_type)
        cls.descriptions = cls.mashup_descriptions + cls.api_descriptions
        cls.tags = cls.mashup_categories + cls.api_categories

