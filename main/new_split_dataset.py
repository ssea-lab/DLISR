# -*- coding:utf-8 -*-
import itertools
import os
import sys
sys.path.append("..")
from main.dataset import dataset, meta_data
import random
import numpy as np
from main.processing_data import process_data
from Helpers.util import read_split_train, read_test_instance, save_split_train, save_test_instance
from Helpers.util import list2dict, save_2D_list, read_2D_list

# 将meta_data.mashup_api_list划分成训练集和测试集
# 然后得到对应的dataset对象

def split_dataset_for_oldScene_KCV(data_dir,num_negatives=6,test_candidates_nums=50, kcv=5):
    name= 'neg_{}_testCandi{}'.format(num_negatives,test_candidates_nums)
    result_path= os.path.join (data_dir,'split_data_oldScene',name)
    mashup_api_list = meta_data.mashup_api_list
    mashup_api_dict = list2dict(mashup_api_list)

    # 返回K个数据对象，每个可以使用相关属性
    test_data_obj = dataset(result_path,name, kcv-1)
    if os.path.exists(test_data_obj.train_instances_path): # 已经划分过
        for i in range(kcv):
            print('has splited data in kcv mode before,read them!')
            data = dataset(result_path,name, i)
            data.read_data() # 从文件中读取对象
            yield data

    else: # 还未划分过
        mashup_ids = list(mashup_api_dict.keys())
        # 首先为每个mashup寻找正负训练用例（训练用的正例也可以当做测试用例
        all_apis = {api_id for api_id in range(meta_data.api_num)}  # 所有api的id

        # 首先为每个mashup制定确定的正负例，候选api等
        # {mid:api_instances}
        mid2true_instances, mid2false_instances, mid2candidate_instances = {}, {}, {}
        for mashup_id, api_ids in mashup_api_dict.items():  # api_ids是set
            unobserved_apis_list = list(all_apis - api_ids)
            random.shuffle(unobserved_apis_list)

            mid2true_instances[mashup_id] = {}
            mid2false_instances[mashup_id] = {}
            mid2candidate_instances[mashup_id] = {}

            api_ids_list = list(api_ids) # 已选择的apis，做正例
            mid2true_instances[mashup_id] = api_ids_list

            all_neg_num = min(meta_data.api_num,num_negatives*len(api_ids_list))
            mid2false_instances[mashup_id] = unobserved_apis_list[:all_neg_num] # 负例

            if test_candidates_nums == 'all': # 选取全部api做测试
                mid2candidate_instances[mashup_id] = list(all_apis)
            else: # 选取部分作为测试，实际组件api和部分unobserved
                mid2candidate_instances[mashup_id] = api_ids_list+unobserved_apis_list[:test_candidates_nums]

        random.shuffle(mashup_ids)
        batch = len(mashup_ids) // kcv
        for i in range(kcv):  # 每个kcv
            start_index = i * batch
            batch_stopindex = len(mashup_ids) if i == kcv - 1 else (i + 1) * batch
            test_mashups= mashup_ids[start_index:batch_stopindex]
            train_mashups = mashup_ids[:start_index] + mashup_ids[batch_stopindex:-1]

            test_mashup_ids, all_candidate_api_ids, all_ground_api_ids = [], [], []
            train_mashup_api_list, train_slt_ids, train_labels = [], [], []

            for mashup_id in train_mashups:
                for true_api_id in mid2true_instances[mashup_id]:
                    train_mashup_api_list.append((mashup_id, true_api_id))  # mashup_id,train_api_id,target
                    train_labels.append(1)
                for false_api_id in mid2false_instances[mashup_id]:
                    train_mashup_api_list.append((mashup_id, false_api_id))  # mashup_id,train_api_id,target
                    train_labels.append(0)

            # test和train格式不同
            # test mashup和api的一行list是多个测试样本,而all_ground_api_ids,test_slt_ids的一行对应前者的一行
            for mashup_id in test_mashups:
                candidate_api_ids = mid2candidate_instances[mashup_id]
                all_candidate_api_ids.append(candidate_api_ids)
                test_mashup_ids.append([mashup_id] * len(candidate_api_ids))
                all_ground_api_ids.append(mid2true_instances[mashup_id])  # 训练用的正例

            train_mashup_id_instances, train_api_id_instances = zip(*train_mashup_api_list)

            data= dataset(result_path,name, i)
            data.set_data_oldScene(train_mashup_id_instances, train_api_id_instances, train_labels, test_mashup_ids, all_candidate_api_ids, all_ground_api_ids)
            print('{}/{} dataset, build done!'.format(i,kcv))
            yield data

def split_dataset_for_newScene_New_KCV(data_dir,num_negatives=6,slt_num=3, combination_num=3, train_positive_samples=50,
          test_candidates_nums=50, kcv=5):
    """

    :param data_dir: 要划分数据的路径
    :param num_negatives: 负采样比例
    :param slt_num: 指定的最大已选择服务的数目
    :param combination_num: 已选择服务中,只选取一部分组合,缓解数据不平衡问题
    :param train_positive_samples: 每个训练用的mashup,除了已选服务，剩下的保留多少个训练正例，防止组件太多的mashup所占的比例太大
    :param test_candidates_nums: 每个mashup要评价多少个待测item
    :param kcv:
    :return: 各个kcv上生成的dataset对象
    """
    name= 'neg_{}_sltNum{}_com_{}_trainPos_{}_testCandi{}'.format(num_negatives,slt_num,combination_num,train_positive_samples,test_candidates_nums)
    result_path= os.path.join (data_dir,'split_data_newScene',name)

    # 返回K个dataset对象，每个可以使用相关属性
    test_data_obj = dataset(result_path,name, kcv-1)
    if os.path.exists(test_data_obj.train_instances_path): # 已经划分过
        for i in range(kcv):
            print('has splited data in kcv mode before,read them!')
            data = dataset(result_path,name, i)
            data.read_data() # 从文件中读取对象
            yield data
    else: # 还未划分过
        # 未划分数据集
        mashup_api_list = meta_data.mashup_api_list
        mashup_api_dict = list2dict(mashup_api_list)

        mashup_ids = list(mashup_api_dict.keys())
        # 首先为每个mashup寻找正负训练用例（训练用的正例也可以当做测试用例
        all_apis = {api_id for api_id in range(meta_data.api_num)}  # 所有api的id

        # 1.首先为每个mashup指定确定的已选服务和对应的正负例/待测api等
        # {mid:{slt_aid_list:api_instances}
        mid2true_instances, mid2false_instances, mid2candidate_instances = {}, {}, {}
        for mashup_id, api_ids in mashup_api_dict.items():  # api_ids是set
            unobserved_apis_list = list(all_apis - api_ids)
            random.shuffle(unobserved_apis_list)

            mid2true_instances[mashup_id] = {}
            mid2false_instances[mashup_id] = {}
            mid2candidate_instances[mashup_id] = {}

            api_ids_list = list(api_ids)
            max_slt_num = min(slt_num, len(api_ids_list) - 1)  # eg:最大需要三个，但是只有2个services构成
            for act_slt_num in range(max_slt_num):  # 选择1个时，两个时...
                act_slt_num += 1
                combinations = list(itertools.combinations(api_ids_list, act_slt_num))
                if combination_num != 'all':  # 只选取一部分组合,缓解数据不平衡问题
                    combination_num = min(len(combinations), combination_num)
                    combinations = combinations[:combination_num]

                for slt_api_ids in combinations:  # 随机组合已选择的api，扩大数据量 # 组合产生,当做已选中的apis
                    train_api_ids = list(api_ids - set(slt_api_ids))  # masked observed interaction 用于训练或测试的

                    if train_positive_samples != 'all':  # 选择一部分正例 做训练或测试
                        train_positive_samples_num = min(len(train_api_ids), train_positive_samples) # 最多50个，一般没有那么多
                        train_api_ids = train_api_ids[:train_positive_samples_num]

                    mid2true_instances[mashup_id][slt_api_ids] = train_api_ids  # 训练用正例 slt_api_ids是tuple

                    num_negative_instances = min(num_negatives * len(train_api_ids), len(unobserved_apis_list))
                    mid2false_instances[mashup_id][slt_api_ids] = unobserved_apis_list[:num_negative_instances]  # 随机选择的负例

                    if test_candidates_nums == 'all':  # 待预测
                        test_candidates_list = list(all_apis - set(slt_api_ids))
                    else:
                        test_candidates_list = unobserved_apis_list[:test_candidates_nums] + train_api_ids
                    mid2candidate_instances[mashup_id][slt_api_ids] = test_candidates_list

        random.shuffle(mashup_ids)
        batch = len(mashup_ids) // kcv
        # 2.然后，根据上面的结果划分为各个KCV，训练和测试
        for i in range(kcv):  # 每个kcv
            start_index = i * batch
            batch_stopindex = len(mashup_ids) if i == kcv - 1 else (i + 1) * batch
            test_mashups= mashup_ids[start_index:batch_stopindex]
            train_mashups = mashup_ids[:start_index] + mashup_ids[batch_stopindex:-1]

            test_mashup_ids, all_candidate_api_ids, test_slt_ids, all_ground_api_ids = [], [], [], []
            train_mashup_api_list, train_slt_ids, train_labels = [], [], []

            for mashup_id in train_mashups:
                for slt_api_ids, true_api_instances in mid2true_instances[mashup_id].items():
                    for true_api_id in true_api_instances:
                        train_slt_ids.append(slt_api_ids)
                        train_mashup_api_list.append((mashup_id, true_api_id))  # mashup_id,train_api_id,target
                        train_labels.append(1)
                for slt_api_ids, false_api_instances in mid2false_instances[mashup_id].items():
                    for false_api_id in false_api_instances:
                        train_slt_ids.append(slt_api_ids)
                        train_mashup_api_list.append((mashup_id, false_api_id))  # mashup_id,train_api_id,target
                        train_labels.append(0)

            # test和train格式不同：
            # train是一行一个样本；
            # test mashup和api的一行list是多个测试样本,而all_ground_api_ids,test_slt_ids的一行对应前者的一行(共用相同的)
            for mashup_id in test_mashups:
                for slt_api_ids, candidate_api_instances in mid2candidate_instances[mashup_id].items():
                    all_candidate_api_ids.append(candidate_api_instances)
                    test_mashup_ids.append([mashup_id] * len(candidate_api_instances))
                    all_ground_api_ids.append(mid2true_instances[mashup_id][slt_api_ids])  # 训练用的正例
                    test_slt_ids.append(slt_api_ids)

            train_mashup_id_instances, train_api_id_instances = zip(*train_mashup_api_list)

            # 3.根据训练集和测试集的划分，初始化dataset对象!!!
            data= dataset(result_path,name, i)
            data.set_data(train_mashup_id_instances, train_api_id_instances, train_labels, train_slt_ids, test_mashup_ids, all_candidate_api_ids, all_ground_api_ids, test_slt_ids)
            print('{}/{} dataset, build done!'.format(i,kcv))
            yield data

    print('you  have splited and saved them!')