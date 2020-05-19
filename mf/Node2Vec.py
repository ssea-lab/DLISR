import argparse
import os
import pickle

import numpy as np
import networkx as nx
import random
from queue import Queue
from gensim.models import Word2Vec
# copy from https://github.com/aditya-grover/node2vec/tree/master/src

class Graph ():
    def __init__(self, nx_G, is_directed, p, q,node2vec_path):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.obj_path= os.path.join(node2vec_path,'graph.obj')

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len (walk) < walk_length:
            cur = walk[-1]
            # print('start:', cur)
            cur_nbrs = sorted (G.neighbors (cur))
            # print('neighbors:',cur_nbrs)
            if len (cur_nbrs) > 0:
                if len (walk) == 1:
                    next = cur_nbrs[alias_draw (alias_nodes[cur][0], alias_nodes[cur][1])]# 第一个点的话，不用考虑上个节点是什么，直接用点-邻居的概率分布抽样
                else:
                    prev = walk[-2]
                    # print('prev,cur:',prev, cur)
                    # print('prev,alias_edges[(prev, cur)]:',alias_edges[(prev, cur)])
                    index = alias_draw (alias_edges[(prev, cur)][0],alias_edges[(prev, cur)][1])
                    # print('chose i-th neighbor:',index)
                    next = cur_nbrs[index] # 从之前的点来的话，要考虑回去的问题，使用论文中的概率采样（这里记做了边的采样）
                # print('next',next)
                walk.append (next)
            else:
                break
        # print(walk)
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list (G.nodes ())
        print ('Walk iteration:')
        for walk_iter in range (num_walks):
            print (str (walk_iter + 1), '/', str (num_walks))
            random.shuffle (nodes)
            for node in nodes: # 每个节点作为出发点采样一次
                walks.append (self.node2vec_walk (walk_length=walk_length, start_node=node))

        return walks

    def update_walks(self, num_walks, walk_length,new_mashup_node,new_api_nbrs=None):
        """
        对于新加的mashup节点和边，随机游走时选择开始节点：1.mashup开始的所有路径 2.mashup开始K跳内的点作为出发点的，且包含mashup的路径
        :param num_walks:
        :param walk_length:
        :param new_mashup_node:
        :return:
        """
        K_hop=5
        walks = []
        # # bfs找潜在近邻出发点
        # start_nodes = bfs(self.G,new_mashup_node,K_hop) # 新节点5跳内的邻居
        # print('nbs in 5 hops:',start_nodes)
        # other_nodes = start_nodes[1:]
        # 简单的，只从邻居出发
        other_nodes = new_api_nbrs
        # print ('Walk iteration:',new_mashup_node,new_api_nbrs)
        for walk_iter in range (num_walks):
            # print (str (walk_iter + 1), '/', str (num_walks))
            walks.append(self.node2vec_walk(walk_length=walk_length, start_node=new_mashup_node)) # 新节点作为出发点的路径
            # print('a')

            random.shuffle (other_nodes)
            for node in other_nodes: # 其他节点作为出发点的路径需要包含新节点
                # print('b')
                temp_walk = self.node2vec_walk (walk_length=walk_length, start_node=node)
                if new_mashup_node in set(temp_walk):
                    walks.append(temp_walk)
                    # print(temp_walk)
        # print(walks)
        return walks

    def get_alias_edge(self, src, dst): # src是来源节点，dst是当前节点
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted (G.neighbors (dst)): # 下个节点
            if dst_nbr == src: # 回到源节点
                unnormalized_probs.append (G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge (dst_nbr, src): # 源节点的其他邻居
                unnormalized_probs.append (G[dst][dst_nbr]['weight'])
            else: # DFS,新加mashup后要更新邻居service的这个值
                unnormalized_probs.append (G[dst][dst_nbr]['weight'] / q)
        norm_const = sum (unnormalized_probs)
        normalized_probs = [float (u_prob) / norm_const for u_prob in unnormalized_probs] # 归一化后的概率分布
        # print('get_alias_edge, before, normalized_probs:', normalized_probs)
        results = alias_setup (normalized_probs)
        # print('get_alias_edge, after, results:', results)
        return results

    def preprocess_transition_probs(self): # 获取节点和边
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes ():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted (G.neighbors (node))]
            norm_const = sum (unnormalized_probs)
            normalized_probs = [float (u_prob) / norm_const for u_prob in unnormalized_probs] # 一个点的邻居，归一化的权重
            alias_nodes[node] = alias_setup (normalized_probs) # 每个节点相应的别名采样的基础

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges ():
                alias_edges[edge] = self.get_alias_edge (edge[0], edge[1])
        else:
            for edge in G.edges ():
                alias_edges[edge] = self.get_alias_edge (edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge (edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

    def update_transition_probs(self,new_mashup_node,new_api_nbrs):
        # 根据新mashup和已选择的apis更新图，图上的边，为随机游走做基础
        # print('new nodes and nbrs:',new_mashup_node,new_api_nbrs) #!!!
        G = self.G
        update_nodes = list(new_api_nbrs)
        update_nodes.append(new_mashup_node)
        for node in update_nodes:
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]  # 一个点的邻居，归一化的权重
            # print('update_transition_probs,before alias_setup, normalized_probs:',normalized_probs)
            self.alias_nodes[node] = alias_setup(normalized_probs)  # 每个节点相应的别名采样的基础
            # print('update_transition_probs,after alias_setup, normalized_probs:', self.alias_nodes[node])

        # 要更新的边,默认无向边(注：无向边的也不同，跟近邻有关)
        new_edges = [(new_mashup_node,api_nbr_node) for api_nbr_node in new_api_nbrs]
        for edge in new_edges:
            self.alias_edges[edge] = self.get_alias_edge (edge[0], edge[1])
        new_edges = [(api_nbr_node, new_mashup_node) for api_nbr_node in new_api_nbrs]
        for edge in new_edges:
            self.alias_edges[edge] = self.get_alias_edge (edge[0], edge[1])

def bfs(G,node,K_hop=5):
    queue = Queue()
    nodeSet = set()
    nodelist=[]
    queue.put(node)
    queue.put('') #终止符
    hop = 0
    nodeSet.add(node) # 判断是否存在
    nodelist.append(node) # 结果
    while not queue.empty():
        cur_node = queue.get()               # 弹出元素
        if cur_node=='': # 每一跳的终止符
            queue.put('')
            hop+=1
            if hop==K_hop:
                break
            else:
                continue
        for next_node in G.neighbors(cur_node):          # 遍历元素的邻接节点
            if next_node not in nodeSet:     # 若邻接节点没有入过队，加入队列并登记
                nodeSet.add(next_node)
                nodelist.append(node)
                queue.put(next_node)
    return nodelist


def alias_setup(probs): # 根据概率分布生成alias抽样的两个前提输入:J是索引，q是值
    K = len (probs)
    q = np.zeros (K)
    J = np.zeros (K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate (probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append (kk)
        else:
            larger.append (kk)

    while len (smaller) > 0 and len (larger) > 0:
        small = smaller.pop ()
        large = larger.pop ()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append (large)
        else:
            larger.append (large)

    return J, q


def alias_draw(J, q):
    K = len (J)

    kk = int (np.floor (np.random.rand () * K))
    if np.random.rand () < q[kk]: # 均匀分布时随便选
        return kk
    else:
        return J[kk]

"""
def parse_args():
    parser = argparse.ArgumentParser (description="Run node2vec.")

    parser.add_argument ('--input', nargs='?', default='graph/karate.edgelist',
                         help='Input graph path')

    parser.add_argument ('--output', nargs='?', default='emb/karate.emb',
                         help='Embeddings path')

    parser.add_argument ('--dimensions', type=int, default=128,
                         help='Number of dimensions. Default is 128.')

    parser.add_argument ('--walk-length', type=int, default=80,
                         help='Length of walk per source. Default is 80.')

    parser.add_argument ('--num-walks', type=int, default=10,
                         help='Number of walks per source. Default is 10.')

    parser.add_argument ('--window-size', type=int, default=10,
                         help='Context size for optimization. Default is 10.')

    parser.add_argument ('--iter', default=1, type=int,
                         help='Number of epochs in SGD')

    parser.add_argument ('--workers', type=int, default=8,
                         help='Number of parallel workers. Default is 8.')

    parser.add_argument ('--p', type=float, default=1,
                         help='Return hyperparameter. Default is 1.')

    parser.add_argument ('--q', type=float, default=1,
                         help='Inout hyperparameter. Default is 1.')

    parser.add_argument ('--weighted', dest='weighted', action='store_true',
                         help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument ('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults (weighted=False)

    parser.add_argument ('--directed', dest='directed', action='store_true',
                         help='Graph is (un)directed. Default is undirected.')
    parser.add_argument ('--undirected', dest='undirected', action='store_false')
    parser.set_defaults (directed=False)

    return parser.parse_args ()
"""


class args(object):
    def __init__(self,node2vec_path):
        """
        :param node2vec_path: 存放node2vec各种结果的路径
        """
        self.node2vec_path=node2vec_path # '../coldstart/node2vec'
        self.input=os.path.join(node2vec_path,'UI.txt')
        self.m_id_map_path= os.path.join(node2vec_path,'m_id_map.csv')#
        self.a_id_map_path = os.path.join(node2vec_path,'a_id_map.csv')  #
        self.output=os.path.join(node2vec_path,'result.txt')
        self.m_embedding=os.path.join(node2vec_path,'m_embedding.txt')
        self.a_embedding = os.path.join(node2vec_path,'a_embedding.txt')
        self.model_path=os.path.join(node2vec_path,'word2vec.model')

        # chenweishen代码默认参数: walk_length = 10, num_walks = 80, p = 0.25, q = 4, window_size=5, workers=3, iter=5
        # 源代码默认参数           walk_length = 80, num_walks = 10, p = 1, q = 1, workers = 8,window_size=10, iter=1
        self.dimensions=25
        self.walk_length=10 # 10
        self.num_walks=80 # 80
        self.window_size=5 # 5
        self.iter=5 # 5
        self.workers=8
        self.p=0.25 # 0.25
        self.q=4 # 4
        self.weighted = False
        self.directed = False


def read_graph(a_args):
    if a_args.weighted:
        G = nx.read_edgelist (a_args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph ())
    else:
        G = nx.read_edgelist (a_args.input, nodetype=int, create_using=nx.DiGraph ())
        for edge in G.edges ():
            G[edge[0]][edge[1]]['weight'] = 1 # value是一个字典？权重

    if not a_args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(a_args,walks,model_path):
    walks = [list(map (str, walk)) for walk in walks] # 把数值转化为''
    model = Word2Vec (walks, size=a_args.dimensions, window=a_args.window_size, min_count=0, sg=1, workers=a_args.workers,
                      iter=a_args.iter)
    model.save(model_path) # word2vec模型对象，可加载
    model.wv.save_word2vec_format(a_args.output)
    return


def call_node2vec(a_args):
    nx_G = read_graph (a_args)
    G_obj = Graph (nx_G, a_args.directed, a_args.p, a_args.q,a_args.node2vec_path)
    G_obj.preprocess_transition_probs()
    walks = G_obj.simulate_walks (a_args.num_walks, a_args.walk_length)
    learn_embeddings (a_args,walks,a_args.model_path) # 存储word2vec模型对象
    with open(G_obj.obj_path, 'wb') as f: # 图对象
        pickle.dump(G_obj,f)
        print('save trainSet_node2vec_embeddings, done!')


def online_learn_embeddings(a_args,walks,model_path,new_mashup_node):
    walks = [list(map(str, walk)) for walk in walks]
    print('online walks for a new mashup_node:',len(walks))
    model = Word2Vec.load(model_path)
    model.build_vocab(sentences=walks, update=True)
    model.train(walks, total_examples=len(walks), epochs=a_args.iter)
    vec = model.wv[str(new_mashup_node)]
    print(vec)
    return vec


def get_newNode_vec(G_obj,a_args,new_mashup_node,new_api_nbrs): # 都是内部索引形式
    G_obj.update_transition_probs(new_mashup_node,new_api_nbrs)
    new_walks = G_obj.update_walks(a_args.num_walks, a_args.walk_length,new_mashup_node,new_api_nbrs)
    new_vec = online_learn_embeddings(a_args,new_walks,a_args.model_path,new_mashup_node)
    return new_vec


if __name__ == "__main__":
    """
    args = parse_args ()
    call_node2vec (args)
    """