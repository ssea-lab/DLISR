# encoding:utf8
import csv

import numpy as np
from numpy.linalg import linalg


def write_mashup_api_pair(mashup_api_pairs, data_path, manner):
    # 存储 mashup api 关系对
    with open(data_path, 'w+') as f:
        if manner == 'list':
            for mashup_id, api_id in mashup_api_pairs:
                f.write("{}\t{}\n".format(mashup_id, api_id))
        elif manner == 'dict':
            for mashup_id, api_ids in mashup_api_pairs.items():
                for api_id in api_ids:
                    f.write("{}\t{}\n".format(mashup_id, api_id))
        else:
            pass


def list2dict(train):
    """
    将（UID，iID）形式的数据集转化为dict  deepcopy
    :param train:
    :return:
    """
    a_dict = {}
    for mashup_id, api_id in train:
        if mashup_id not in a_dict.keys():
            a_dict[mashup_id] = set()
        a_dict[mashup_id].add(api_id)
    return a_dict


def dict2list(train):
    _list=[]
    for mashup,apis in train:
        for api in apis:
            _list.append(mashup,api)
    return _list

def cos_sim(A, B):
    if isinstance(A, list):
        A = np.array(A)
        B = np.array(B)
    num = float((A * B).sum())  # 若为列向量则 A.T  * B
    denom = linalg.norm(A) * linalg.norm(B)
    cos = num / denom  if denom!=0 else 0 # 余弦值
    return cos

def Euclid_sim():
    pass

def transform_dict(a_dict):
    new_dict={}
    for key,value in a_dict.items():
        new_dict[value]=key
    return new_dict


def get_id2index(doc_path):  # librc处理后的id2index文件
    id2index = {}
    reader = csv.DictReader(open(doc_path, 'r'))  # r
    for row in reader:
        id2index[int(row['id'])] = int(row['index'])
    return id2index


def save_id2index(id2index,doc_path):
    with open(doc_path,'w') as f:
        f.write('id,index\n')
        for id,index in id2index.items():
            f.write('{},{}\n'.format(id,index))

def read_2D_list(path):
    _list = []
    with open (path, 'r') as f:
        line = f.readline ()
        while line is not None:
            ids = [int (str_id) for str_id in line.split ()]
            if len (ids) == 0:
                break
            _list.append (ids)
            line = f.readline ()
    return _list


def save_2D_list(path, _list,mode='w'):
    with open (path, mode) as f:
        for index in range (len (_list)):
            f.write (' '.join ([str (id) for id in _list[index]]))
            f.write ('\n')


def save_split_train(path, train_test_mashup_api_list, train_labels=None):
    """
    用在保存split得到的 train，test结构（list)；也可用于保存据此生成的有label的实例
    格式：mashup_id api_id label(可选）
    :param path:
    :param train_test_mashup_api_list:
    :param train_labels:
    :return:
    """
    with open (path, 'w') as f:
        if train_labels is None:
            for mashup_id, api_id in train_test_mashup_api_list:
                f.write ('{} {}\n'.format (mashup_id, api_id))
        else:
            assert len (train_test_mashup_api_list) == len (train_labels)
            for i in range (len (train_test_mashup_api_list)):
                a_pair = train_test_mashup_api_list[i]
                f.write ('{} {} {}\n'.format (a_pair[0], a_pair[1], train_labels[i]))


def read_split_train(path, have_label):
    mashup_api_list = []
    labels = []
    with open (path, 'r') as f:
        line = f.readline ()
        while line is not None:
            ids = [int (str_id) for str_id in line.split ()]
            if len (ids) == 0:
                break
            mashup_api_list.append ((ids[0], ids[1]))
            if have_label:
                labels.append (ids[2])
            line = f.readline ()
    if have_label:
        return mashup_api_list, labels
    else:
        return mashup_api_list


def save_test_instance(path, test_mashup_id_instances, test_api_id_instances):
    """
    将实际使用的测试样例 写入文件 ： mashup_id api_id1 api_id2
    :param path:
    :param test_mashup_id_instances:
    :param test_api_id_instances:
    :return:
    """
    with open (path, 'w') as f:
        assert len (test_mashup_id_instances) == len (test_api_id_instances)
        for index in range (len (test_mashup_id_instances)):
            mashup_id = test_mashup_id_instances[index][0]
            api_ids = test_api_id_instances[index]
            f.write (str (mashup_id) + ' ')
            f.write (' '.join ([str (api_id) for api_id in api_ids]))
            f.write ('\n')


def read_test_instance(test_instance_path):
    test_mashup_id_instances = []
    test_api_id_instances = []
    with open (test_instance_path, 'r') as f:
        line = f.readline ()
        while line is not None:
            ids = [int (str_id) for str_id in line.split ()]
            if len (ids) == 0:
                break
            mashup_id = ids[0]
            api_ids = ids[1:]
            test_mashup_id_instances.append ([mashup_id] * len (api_ids))
            test_api_id_instances.append (api_ids)

            line = f.readline ()
    return test_mashup_id_instances, test_api_id_instances

if __name__=='__main__':
    a=[1,0]
    b=[-1,0]
    print(cos_sim(a,b))