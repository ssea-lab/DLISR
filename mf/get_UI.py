import os
import pickle
import sys
sys.path.append("..")
import numpy as np

from Helpers.util import get_id2index, save_id2index,write_mashup_api_pair
from mf.Node2Vec import call_node2vec, args, get_newNode_vec


class get_UV():
    # 得到一个封装UV结果的对象
    def __init__(self,data_path, mode, train_mashup_api_list, slt_num = 0):
        """
        外部获取UI矩阵的接口类,  训练集中mashup,api  按照id从小到大的顺序输出，UI矩阵的顺序与之对应
        :param data_path: 对哪种划分
        :param mode: 使用哪种矩阵分解方法
        :param train_mashup_api_list: 训练集
        :return:
        """
        if slt_num == 0:
            MF_path=os.path.join(data_path,'U_V',mode)  # eg:C:\Users\xiaot\Desktop\MF+CNN\GX\data\split_data\cold_start\U_V\pmf\
        else:
            MF_path = os.path.join(data_path, 'U_V', str(slt_num)+'_'+mode) # 1_BPR
        self.MF_path = MF_path
        if not os.path.exists(MF_path):
            os.makedirs(MF_path)
            print(MF_path+' not exits,and created one!')
        self.m_ids,self.a_ids=zip(*train_mashup_api_list)
        self.m_ids=np.unique(self.m_ids)
        self.a_ids=np.unique(self.a_ids)

        self.m_embeddings, self.a_embeddings =None,None
        if mode=='node2vec':
            self.m_embeddings, self.a_embeddings=get_UV_from_Node2vec(MF_path,train_mashup_api_list)

        elif mode=='BiNE':
            print(MF_path)
            rating_train_path=os.path.join(MF_path,'rating_train.dat')
            if not os.path.exists(rating_train_path):
                prepare_data_for_BiNE(train_mashup_api_list,rating_train_path)
                print('you ought to run BiNE first!!!')
                sys.exit()
                # 有时间将BiNE的代码整合到该工程中
            self.m_embeddings, self.a_embeddings = get_BiNE_UI_embeddings(os.path.join(MF_path,'vectors_u.dat')),get_BiNE_UI_embeddings(os.path.join(MF_path,'vectors_v.dat'))
        else:
            self.m_embeddings=get_UV_from_librec(MF_path, "mashup", self.m_ids)
            self.a_embeddings=get_UV_from_librec(MF_path, "api", self.a_ids)

        self.a_id2index = {id: index for index, id in enumerate (self.a_ids)}
        self.m_id2index = {id: index for index, id in enumerate (self.m_ids)}

        self.m_sltApis_nodeEmb_map_path = os.path.join(self.MF_path, 'online_embedding.map')
        if os.path.exists(self.m_sltApis_nodeEmb_map_path ):
            self.read_onlineNode2vec()
            print('load learned node2vec,done!')
        else:
            self.m_sltApis_nodeEmb_map = {} # 新处理，由新mashup节点，已选择服务，在线生成新mashup的节点表示

    def save_onlineNode2vec(self): # online时一折上实验完成后要保存
        with open(self.m_sltApis_nodeEmb_map_path, 'wb') as f:
            pickle.dump(self.m_sltApis_nodeEmb_map, f)
            print('save onlineNode2vec result,done!')

    def read_onlineNode2vec(self):
        with open(self.m_sltApis_nodeEmb_map_path, 'rb') as f:
            self.m_sltApis_nodeEmb_map = pickle.load(f)

        # # 归一化
        # original_values = np.array(list(self.m_sltApis_nodeEmb_map.values()))
        # original_values -= np.mean(original_values, axis=0)
        # original_values /= np.std(original_values, axis=0)
        # self.m_sltApis_nodeEmb_map = dict(zip(self.m_sltApis_nodeEmb_map.keys(),original_values))

def get_UV_from_librec(MF_path, user_or_item, ordered_ids):
    """
    返回从librec得到的结果，按照id大小排列
    :param MF_path:
    :param user_or_item:
    :param ordered_ids: 一般是按照mashup，api的id从小到大排列的
    :return:
    """
    if user_or_item=="mashup":
        id2index_path=MF_path + "/userIdToIndex.csv"
        matrix_path=MF_path+"/U.txt"
    elif user_or_item == "api":
        id2index_path = MF_path + "/itemIdToIndex.csv"
        matrix_path = MF_path + "/V.txt"

    matrix=np.loadtxt(matrix_path)
    id2index=get_id2index(id2index_path)
    ordered_numpy=np.array([matrix[id2index[id]] for id in ordered_ids])
    return ordered_numpy


def prepare_data_for_Node2vec(a_args,train_mashup_api_list):
    """
    :param train_mashup_api_list: # 需传入内部索引？？？外部
    :return:
    """
    m_ids,a_ids=zip(*train_mashup_api_list)
    m_ids=np.unique(m_ids)
    a_ids=np.unique(a_ids)
    m_num=len(m_ids)

    # 对mashup和api的id进行统一
    m_id2index={m_ids[index]:index+1 for index in range(len(m_ids))}
    save_id2index (m_id2index, a_args.m_id_map_path)

    a_id2index = {a_ids[index]:m_num+index + 1 for index in range(len (a_ids))}
    save_id2index(a_id2index,a_args.a_id_map_path)

    pair=[]
    for m_id,a_id in train_mashup_api_list:
        pair.append((m_id2index[m_id],a_id2index[a_id])) # 内部索引
    write_mashup_api_pair(pair,a_args.input,'list')
    print('prepare_data_for_Node2vec,done!')


def get_UV_from_Node2vec(node2vec_path,train_mashup_api_list):
    """
    传入U-I,返回mashup和api的embedding矩阵,按照id大小排列
    :param node2vec_path:
    :param train_mashup_api_list:
    :return:
    """
    a_args= args(node2vec_path)
    if not os.path.exists(a_args.m_embedding):
        prepare_data_for_Node2vec(a_args,train_mashup_api_list)
        call_node2vec(a_args)

        m_ids,a_ids=zip(*train_mashup_api_list)
        m_ids = np.unique (m_ids)
        a_ids = np.unique (a_ids)

        index2embedding={}
        with open(a_args.output, 'r') as f:
            line=f.readline() # 第一行是信息
            line =f.readline()
            while(line):
                l=line.split(' ')
                index=int(l[0])
                embedding=[float(value) for value in l[1:]]
                index2embedding[index]=embedding

                line = f.readline ()

        m_id2index=get_id2index(a_args.m_id_map_path)
        a_id2index = get_id2index (a_args.a_id_map_path)

        m_embeddings=[]
        a_embeddings=[]
        # 按照id大小输出，外部使用
        for m_id in m_ids:
            m_embeddings.append(index2embedding[m_id2index[m_id]])
        for a_id in a_ids:
            a_embeddings.append(index2embedding[a_id2index[a_id]])

        np.savetxt(a_args.m_embedding, m_embeddings)
        np.savetxt (a_args.a_embedding, a_embeddings)
        return np.array(m_embeddings),np.array(a_embeddings)
    else:
        m_embeddings=np.loadtxt(a_args.m_embedding)
        a_embeddings=np.loadtxt (a_args.a_embedding)
        return m_embeddings,a_embeddings


def update_graph(a_args,new_mashup_node,new_api_nbrs):
    G_obj_path = os.path.join(a_args.node2vec_path,'graph.obj')
    with open(G_obj_path, 'rb') as f: # 训练集上的用于训练的图对象(G_obj.G才是networkx中的graph)，一次生成永久保存；线上的更新都基于该对象
        G_obj = pickle.load(f)
    m_id2index = get_id2index(a_args.m_id_map_path)
    a_id2index = get_id2index(a_args.a_id_map_path)
    new_mashup_node_index = len(m_id2index)+len(a_id2index)+1 # 训练编码时从1开始
    G_obj.G.add_node(new_mashup_node_index)
    avai_api_nbrs_index = []
    for api_nbr_id in new_api_nbrs:
        if api_nbr_id not in a_id2index.keys(): # 如果使用的api节点不在训练集中，跳过
            continue
        else:
            api_nbr_index = a_id2index[api_nbr_id]
            avai_api_nbrs_index.append(api_nbr_index)
            G_obj.G.add_edge(new_mashup_node_index,api_nbr_index)
            G_obj.G[new_mashup_node_index][api_nbr_index]['weight'] = 1 # 默认权重1

            G_obj.G.add_edge(api_nbr_index,new_mashup_node_index)
            G_obj.G[api_nbr_index][new_mashup_node_index]['weight'] = 1 # 默认权重1

    if not a_args.directed: # 转化为无向图
        G_obj.G = G_obj.G.to_undirected()
    return G_obj,new_mashup_node_index,avai_api_nbrs_index


def get_newNodeVec_from_Node2vec(get_UV_obj,node2vec_path,test_mashup_list,test_slt_ids): # 传入测试集的mashup和已选择的api, vector
    """

    :param get_UV_obj: UV对象，其m_sltApis_nodeEmb_map存储(m_id, slt_apis_tuple)到embedding向量的映射。该函数处理后，该对象新增新的映射
    :param node2vec_path:
    :param test_mashup_list: 测试集的mashup列表(部分)
    :param test_slt_ids: 测试集的新mashup已选择的服务列表(部分)
    :return:
    """
    new_vecs = []
    a_args = args(node2vec_path)
    for i in range(len(test_mashup_list)):
        m_id = test_mashup_list[i]
        slt_apis_tuple = tuple(test_slt_ids[i])
        _key = (m_id, slt_apis_tuple)
        if _key not in get_UV_obj.m_sltApis_nodeEmb_map.keys():
            print('mashup node and api_nbrs, id:',m_id, slt_apis_tuple)
            G_obj,new_mashup_node_index,new_api_nbrs_indexes = update_graph(a_args,m_id,slt_apis_tuple) # 将新点转化为局部索引，添加新边，加到图中
            print('mashup node and api_nbrs, index:', new_mashup_node_index, new_api_nbrs_indexes)
            if len(new_api_nbrs_indexes)==0: # 该新节点调用过的api在历史中均未出现过
                new_vec = np.random.rand(25)
            else:
                print('selected avaliable apis!')
                new_vec = get_newNode_vec(G_obj,a_args,new_mashup_node_index,new_api_nbrs_indexes)
            get_UV_obj.m_sltApis_nodeEmb_map[_key] = new_vec
        else:
            new_vec = get_UV_obj.m_sltApis_nodeEmb_map[_key]
        new_vecs.append(new_vec)
    return new_vecs


def prepare_data_for_BiNE(train_mashup_api_list,result_path):
    with open(result_path,'w') as f:
        for m_id,a_id in train_mashup_api_list:
            f.write ("u{}\ti{}\t{}\n".format(m_id,a_id,1))


def get_BiNE_UI_embeddings(path):
    id2embeddings={}
    with open (path,'r') as f:
        line = f.readline ()
        while line:
            a_line = line.strip().split (" ")
            id=int(a_line[0][1:])
            embedding = [float(value) for value in a_line[1:]]
            id2embeddings[id]= embedding
            line = f.readline ()
    sorted_id2embeddings_list =sorted(id2embeddings.items (), key= lambda x:x[0])
    sorted_ids,sorted_embeddings= zip(*sorted_id2embeddings_list)
    # print(np.array(sorted_embeddings))
    return np.array(sorted_embeddings)



