# -*- coding:utf-8 -*-
import os
# import nltk.data
# from nltk.corpus import stopwords
# from nltk.tokenize import WordPunctTokenizer,word_tokenize
from collections import OrderedDict

from Helpers import util
import pickle
import numpy as np
import math
import csv
from main.new_para_setting import new_Para

doc_sparator = ' >>\t'

punctuation = '[<>/\s+\.\!\/_,;:$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+' # 文本处理时去除特殊符号

domain_stopwords = {}  # 专用语料停用词

# 完成了两件事：
# 1.处理原始文件，将内容信息存放在字典中并使用pickle封装；可以根据get_mashup_api_id2info()接口获取信息；也可以通过get_all_texts()直接获得所有mashup和apis的text/tag信息
# 2.将合法的mashup和api编码，生成调用矩阵。可以获取的信息有mashup_num,api_num,mashup_api_list等

class process_data(object):
    mashup_info_result = 'mashup.info'
    mashup_api_matrix_name_path='mashup_api_matrix.name'

    def __init__(self, base_path,preprocess_new=False,min_api_num=2):  # 每个根目录一个对象
        self.base_path = base_path
        self.processed_info_dir=os.path.join(base_path, 'processed_info') # 所有处理过的文件的目录

        if not os.path.exists(self.processed_info_dir):
            os.makedirs(self.processed_info_dir)
        self.processed = os.path.exists(os.path.join(self.processed_info_dir, process_data.mashup_info_result))  # 是否预处理过

        # mashup-api的调用矩阵，name形式，调用次数大于min_api_num的才纳入统计，可以在此基础上进行处理
        # 但 mashup.info中Related APIs字段存储了全部调用信息，跟min_api_num无关
        self.mashup_api_matrix_in_name=None

        self.mashup_name2info={}
        self.api_name2info = {}

        self.min_api_num = min_api_num
        if new_Para.param.if_data_new: # 旧数据集还是新的(旧的效果好)
            self.process_raw_data_NewData(preprocess_new)
        else:
            self.process_raw_data(preprocess_new)

        self.mashup_name2index = {}
        self.api_name2index = {}
        self.mashup_api_list=[] # mashup，api调用关系对，id 形式
        self.mashup_num=0
        self.api_num = 0

        self.encodeIds ()
        if not os.path.exists(os.path.join(self.processed_info_dir,'statistics.csv')):
            self.statistics()

        # mashup_descriptions, api_descriptions, mashup_categories, api_categories = self.get_all_texts()

    def process_raw_data(self,preprocess_new):
        """
        处理爬取的原始文件
        :param min_api_num  mashup和api调用对：只有长度大于2的mashup参与统计
        .info包含所有mashup/api的信息  pickle封装dict
        name-id映射，以及mashup_api_matrix_in_name只有长度大于2的mashup参与统计
        """

        if not self.processed or preprocess_new:
            dirs = [os.path.join(self.base_path, 'Mashup'), os.path.join(self.base_path, 'API')]
            self.mashup_api_matrix_in_name = OrderedDict()
            for data_type, data_dir in enumerate(dirs): # 分别处理mashup和api. 0 mashup 1 api 决定tag（类别）在文件中的位置
                for doc_name in os.listdir(data_dir):
                    name = ''
                    a_dict = {}
                    with open(os.path.join(data_dir, doc_name),encoding='utf-8') as f: #处理时所有mashup和api均写入
                        for line in f.readlines():
                            line = line.split(doc_sparator)

                            if len(line)<2 and 'Description' in a_dict.keys(): # 处理跨行的Description
                                a_dict['Description']=a_dict['Description']+line[0].lower().strip()
                                continue

                            Key=line[0]
                            Value=line[1]
                            if Key == 'Name':
                                name = '-'.join(Value.strip().lower().split()) # name中有空格，转化为跟related apis中相同的格式
                            elif Key == 'Description': # 类别词和描述词要小写
                                a_dict[Key] = Value.lower().strip()
                            elif Key in ['Primary Category','Secondary Categories','Categories']:# 类别词进一步处理，跟文本相同的list形式，方便操作
                                Value=  Value.lower().strip().split(', ') # 多个类别标签使用', '分隔
                                new_Values=[] # 有些标签有多个词汇，空格分隔，现使用-连接
                                for category in Value:
                                    category_words=category.split(' ')
                                    if len(category_words)>1:
                                        category='-'.join(category_words)
                                    new_Values.append(category)
                                a_dict[Key]=new_Values
                            elif Key =="Related APIs":
                                a_dict[Key] = Value
                                if len(Value.split()) >= self.min_api_num:  # 统计计数只考虑长度大于2的
                                    self.mashup_api_matrix_in_name[name] = [api.strip().lower() for api in Value.split()] # Related api中的名称和 name段的经过相同的处理
                            elif Key=="API Provider":
                                a_dict[Key] = Value
                            else:
                                pass
                    if a_dict.get('Description') is not None:
                        a_dict['final_description'] = NLP_tool(a_dict['Description'])

                    # name到对应info的字典；实际上保留了所有mashup和api的全部信息！
                    if data_type == 0:
                        self.mashup_name2info[name] = a_dict
                    elif data_type == 1:
                        self.api_name2info[name] = a_dict

            with open(os.path.join(self.processed_info_dir, 'mashup.info'), 'wb') as file:  # 存储info
                pickle.dump(self.mashup_name2info, file)
            with open(os.path.join(self.processed_info_dir, 'api.info'), 'wb') as file:  # 存储info
                pickle.dump(self.api_name2info, file)
            with open (os.path.join (self.processed_info_dir, process_data.mashup_api_matrix_name_path), 'wb') as file:  # 存储info
                pickle.dump (self.mashup_api_matrix_in_name, file)
            print("process_raw_data, done!")

        else:
            with open (os.path.join(self.processed_info_dir, process_data.mashup_api_matrix_name_path), 'rb') as file:
                self.mashup_api_matrix_in_name = pickle.load (file)
            self.mashup_name2info = self.get_mashup_api_name2info('mashup')
            self.api_name2info = self.get_mashup_api_name2info('api')

    def process_raw_data_NewData(self,preprocess_new,min_api_num=2):
        if not self.processed or preprocess_new:
            data_paths = [os.path.join(self.base_path, 'mashups.csv'), os.path.join(self.base_path, 'apis.csv')]
            # 'Related APIs','Categories'这些都是"['amazon-product-advertising', 'google-o3d']"形式，去除两边的""要eval()
            mashup_domains = ['Name','Description','Related APIs','Categories','Company']
            api_domains = ['Name', 'Description', 'Categories', 'API Provider']
            domains = [mashup_domains,api_domains]

            self.mashup_api_matrix_in_name = {}
            # 0 mashup 1 api 决定tag（类别）在文件中的位置
            for data_type, data_path in enumerate(data_paths):
                head_line = True
                with open(data_path,encoding='utf-8',errors='ignore')as f:
                    f_csv = csv.reader(f)
                    for row in f_csv: # list形式
                        if head_line:
                            head_line = False
                            continue
                        a_dict = {}
                        name = '-'.join(row[0].strip().lower().split()) # 'name'用-连接
                        for index, domain_name in enumerate(domains[data_type][1:]): # 各领域的处理
                            Value = row[index+1]
                            # print(Value)
                            if domain_name =='Description': # 描述词要小写
                                a_dict[domain_name] = Value.lower().strip()
                            elif domain_name == 'Categories':  # 类别词进一步处理，跟文本相同的list形式，方便操作

                                if not Value:
                                    a_dict[domain_name] = []
                                else:
                                    Value = Value.lower() # mashup和API的大小写不同！
                                    Value = eval(Value)
                                    new_Values = []  # 有些标签有多个词汇，空格分隔，现使用-连接
                                    for category in Value:
                                        category_words = category.strip().split(' ')
                                        if len(category_words) > 1:
                                            category = '-'.join(category_words)
                                        new_Values.append(category)
                                    a_dict[domain_name] = new_Values
                            elif domain_name == "Related APIs":
                                if not Value:
                                    a_dict[domain_name] = []
                                else:
                                    if ',' in Value:
                                        Value = eval(Value) # 多个时，"['amazon-product-advertising', 'google-o3d']"
                                    a_dict[domain_name] = Value
                                    if len(Value) >= min_api_num:  # 统计计数只考虑长度大于2的
                                        self.mashup_api_matrix_in_name[name] = [api.strip().lower() for api in Value]  # Related api中的名称和 name段的经过相同的处理
                            elif domain_name == "API Provider" or domain_name == "Company":  # 新增.为空时怎么办??
                                a_dict[domain_name] = Value
                            else:
                                pass

                            if a_dict.get('Description') is not None:
                                a_dict['final_description'] = NLP_tool(a_dict['Description'])

                        if data_type == 0:
                            self.mashup_name2info[name] = a_dict  # name到对应info的字典；实际上保留了所有mashup和api的全部信息！
                        elif data_type == 1:
                            self.api_name2info[name] = a_dict  # name到对应info的字典；实际上保留了所有mashup和api的全部信息！

            with open(os.path.join(self.processed_info_dir, 'mashup.info'), 'wb') as file:  # 存储info
                pickle.dump(self.mashup_name2info, file)
            with open(os.path.join(self.processed_info_dir, 'api.info'), 'wb') as file:  # 存储info
                pickle.dump(self.api_name2info, file)
            with open (os.path.join (self.processed_info_dir, process_data.mashup_api_matrix_name_path), 'wb') as file:  # 存储info
                pickle.dump (self.mashup_api_matrix_in_name, file)
            print("process_raw_data, done!")

        else:
            with open (os.path.join(self.processed_info_dir, process_data.mashup_api_matrix_name_path), 'rb') as file:
                self.mashup_api_matrix_in_name = pickle.load (file)
            self.mashup_name2info = self.get_mashup_api_name2info('mashup')
            self.api_name2info = self.get_mashup_api_name2info('api')

    def encodeIds(self, remain_null_apis=False, remain_field_blank_items=False):
        """
        处理调用关系，重新编码（打id）！！！得到mashup,api的name-id映射，以及id形式的关系对 mashup_api_list
        :param remain_null_apis: 是否保留调用关系中未曾出现过的api？
        :param remain_field_blank_items: 是否保留某个域（text，标签）为空的mashup/api？  False时，一旦一个字段为空，不要这个数据，太严格
        :return:
        """
        self.mashup_name2index = {} # 是调用关系中出现且合法的mashup和api
        self.api_name2index = {}

        def is_valid_item(mashup_or_api, name):
            # 判断一个mashup或者api是否符合要求，要编码处理
            if mashup_or_api == 'api':
                if name in self.api_name2index.keys ():
                    return True
                else:
                    api_info = self.api_name2info.get (name)
                    if not remain_null_apis and api_info is None:
                        return False

                    text = api_info.get ('final_description')
                    p_tag = api_info.get ('Primary Category')
                    s_tag = api_info.get ('Secondary Categories')
                    Categories_tag = api_info.get ('Categories')
                    provider_tag = api_info.get ('API Provider') # !!!!!是否需要排除为空的???

                    if new_Para.param.if_data_new:
                        if not remain_field_blank_items and (text is None or Categories_tag is None):
                            return False
                    else:
                        if not remain_field_blank_items and (text is None or p_tag is None or s_tag is None):
                            return False

            elif mashup_or_api == 'mashup':
                if name in self.mashup_name2index.keys ():
                    return True
                else:
                    mashup_info = self.mashup_name2info.get (name)
                    text = mashup_info.get ('final_description')
                    tags = mashup_info.get ('Categories')
                    if not remain_field_blank_items and (tags is None or text is None):
                        return False
            return True

        if not os.path.exists (os.path.join (self.processed_info_dir, 'mashup_api')): # 未进行过编码
            mashup_index = 1 # 从1开始编码！！！
            api_index = 1 # 从1开始编码！！！
            for mashup_name, api_names in self.mashup_api_matrix_in_name.items():
                if is_valid_item('mashup',mashup_name) and mashup_name not in self.mashup_name2index.keys(): # mashup满足要求
                    # 首先对api进行一次遍历，满足要求的api数目大于2时才认为mashup符合要求
                    valid_api_num=0
                    for api_name in api_names:
                        if is_valid_item('api',api_name):
                            valid_api_num+=1
                    if valid_api_num<self.min_api_num:
                        continue

                    # 一个新的且满足需求的mashup进行编码
                    self.mashup_name2index[mashup_name] = mashup_index

                    for api_name in api_names:
                        if api_name not in self.api_name2index.keys() and is_valid_item('api',api_name):
                            self.api_name2index[api_name] = api_index
                            api_index += 1
                        if api_name in self.api_name2index.keys():
                            self.mashup_api_list.append ((mashup_index,self.api_name2index[api_name]))
                    mashup_index += 1

            with open(os.path.join(self.processed_info_dir, 'mashup_name2index'), 'wb') as file:  # 存储name到id映射
                pickle.dump(self.mashup_name2index, file)
            with open(os.path.join(self.processed_info_dir, 'api_name2index'), 'wb') as file:
                pickle.dump(self.api_name2index, file)

            # 合法mashup和api的数目
            self.mashup_num = len(self.mashup_name2index)
            self.api_num = len(self.api_name2index)
            print("write name2index,done!")
            print("Num of mashup:{},Num of api:{}!".format(self.mashup_num,self.api_num))

            # 存储 mashup api 关系对，所有合法的！
            util.write_mashup_api_pair(self.mashup_api_list, os.path.join(self.processed_info_dir, 'mashup_api'), 'list')
            print("write mashup_api_pair,done!")
        else:
            self.mashup_name2index=self.get_mashup_api_index2name('mashup', index2name=False)
            self.api_name2index = self.get_mashup_api_index2name ('api', index2name=False)
            self.mashup_api_list=self.get_mashup_api_pair('list')
            self.mashup_num = len(self.mashup_name2index)
            self.api_num = len(self.api_name2index)

    def get_mashup_api_name2info(self, mashup_or_api):
        """
        返回mashup/api 的name到info(text/tag)的映射  return 1 dicts:  string->dict
        """
        if not (mashup_or_api == 'mashup' or mashup_or_api == 'api'):
            raise ValueError("must input 'mashup' or 'api' ")
        else:
            final_docname = os.path.join(self.processed_info_dir, mashup_or_api + '.info')  # 文件名为mashup/api

        with open(final_docname, 'rb') as file2:
            return pickle.load(file2)

    def get_mashup_api_index2name(self, mashup_or_api, index2name=True):
        # 返回mashup/api 的名称到index的映射  默认是id-name

        if not (mashup_or_api == 'mashup' or mashup_or_api == 'api'):
            raise ValueError("must input 'mashup' or 'api' ")
        else:
            map_path = mashup_or_api + '_name2index' # +'.csv'

        a_map = {}
        name2index={}

        with open(os.path.join(self.processed_info_dir, map_path), 'rb') as file2:
            name2index=pickle.load(file2)
        if index2name:
            for name,index in name2index.items():
                a_map[index]=name
        return a_map if index2name else name2index

    def get_mashup_api_id2info(self, mashup_or_api):
        # 返回由id直接得到info的dict  用在将关系对和对应的text输入模型
        # 一般作为外部接口，调用了get_mashup_api_name2info和get_mashup_api_index2name
        if not (mashup_or_api == 'mashup' or mashup_or_api == 'api'):
            raise ValueError("must input 'mashup' or 'api' ")

        name2info = self.get_mashup_api_name2info(mashup_or_api)
        index2name = self.get_mashup_api_index2name(mashup_or_api)

        id2info = {}
        for id, name in index2name.items():
            info=name2info.get(name)
            id2info[id]=info if info is not None else {} # 可能一个被mashsup调用的api没有相关信息，此时值为{}! 进而使得不存在信息的api的.get()=None

        return id2info

    def get_mashup_api_pair(self, manner):
        """
        获取全部的关系对：pair list:[(m,a1),(m,a2)]  or  dict{(m:{a1,a2})} value:set!!!
        para:
        manner: 'list' or 'dict'
        """
        if not (manner == 'list' or manner == 'dict'):
            raise ValueError("must input 'list' or 'dict' ")

        a_list = []
        a_dict = {}
        with open(os.path.join(self.processed_info_dir, 'mashup_api'), 'r') as f:
            for line in f.readlines():
                if line is not None:
                    # print(line)
                    line = line.strip().split('\t')
                    m_id = int(line[0])
                    api_id = int(line[1])
                    if manner == 'list':
                        a_list.append((m_id, api_id))
                    if manner == 'dict':
                        if m_id not in a_dict:
                            a_dict[m_id] = set()
                        a_dict[m_id].add(api_id)

        return a_list if manner == 'list' else a_dict

    def get_api_co_vecs(self,pop_mode=''): # pop数值是否规约到0-1？？？
        """
        返回每个api跟所有api的共现次数向量和每个api的popularity
        :return:
        """
        all_api_num=len(self.get_mashup_api_index2name('api'))
        api_co_vecs = np.zeros ((all_api_num,all_api_num),dtype='float32')
        api2pop=np.zeros((all_api_num,),dtype='float32')
        mashup_api_pair=self.get_mashup_api_pair('dict')
        for mashup,apis in mashup_api_pair.items():
            for api1 in apis:
                api2pop[api1]+=1.0
                for api2 in apis:
                    if api1!=api2:
                        api_co_vecs[api1][api2]+=1.0
        if pop_mode=='sigmoid':
            api2pop=[1.0/(1+pow(math.e,-1*pop)) for pop in api2pop]
        return api_co_vecs,api2pop

    def get_all_texts(self,Category_type='all'):
        """
        得到所有mashup api的description和category信息！ 按id(编码的全局ID)排列
        !!!应该成为外部接口，而不是get_mashup_api_field()和get_mashup_api_allCategories()!!!
        :return: 返回的是字符串形式,用' '分隔！！！信息不存在则为''
        """
        mashup_id2info = self.get_mashup_api_id2info('mashup')
        api_id2info = self.get_mashup_api_id2info('api')

        a_Description = get_mashup_api_field(mashup_id2info, 0, 'final_description')
        a_Categories = get_mashup_api_allCategories('mashup', mashup_id2info, 0,Category_type)
        print(a_Description)
        print(a_Categories)
        # mashup_descriptions = [get_mashup_api_field(mashup_id2info, mashup_id, 'final_description') for mashup_id in range(self.mashup_num)]
        # api_descriptions = [get_mashup_api_field(api_id2info, api_id, 'final_description') for api_id in range(self.api_num)]
        # print(mashup_descriptions)
        # print(api_descriptions)

        if  isinstance(a_Description,str):
            mashup_descriptions = [get_mashup_api_field(mashup_id2info, mashup_id, 'final_description') for mashup_id in range(self.mashup_num)]
            api_descriptions = [get_mashup_api_field(api_id2info, api_id, 'final_description') for api_id in range(self.api_num)]
        elif isinstance(a_Description,list):
            mashup_descriptions = [' '.join(get_mashup_api_field(mashup_id2info, mashup_id, 'final_description')) + ' ' for mashup_id in range(self.mashup_num)]
            api_descriptions = [' '.join(get_mashup_api_field(api_id2info, api_id, 'final_description')) + ' ' for api_id in range(self.api_num)]

        mashup_categories = [' '.join(get_mashup_api_allCategories('mashup', mashup_id2info, mashup_id,Category_type))+' ' for mashup_id in range(self.mashup_num)]
        api_categories = [' '.join(get_mashup_api_allCategories('api', api_id2info, api_id,Category_type))+' ' for api_id in range(self.api_num)]

        return mashup_descriptions,api_descriptions,mashup_categories,api_categories

    def statistics(self):
        api_id2info = self.get_mashup_api_id2info('api')

        # 每个mashup构成的api数目
        mashup_api=self.get_mashup_api_pair('dict')
        api_num_per_mashup=sorted([len(apis) for apis in mashup_api.values()]) # eg:222334

        mashup_descriptions, api_descriptions, mashup_categories, api_categories= self.get_all_texts()
        # 每个mashup的文本长度,tag数目
        mashup_description_lens= sorted([str(len(m_des.split(' '))) for m_des in mashup_descriptions])
        mashup_categories_lens = sorted([str(len(m_cate.split(' '))) for m_cate in mashup_categories])
        # 每个api的文本长度,tag数目,provider数目
        api_description_lens= sorted([str(len(a_des.split(' '))) for a_des in api_descriptions])
        api_categories_lens = sorted([str(len(a_cate.split(' '))) for a_cate in api_categories])
        api_provider_lens= sorted([str(len(get_mashup_api_field(api_id2info,api_id,'API Provider'))) for api_id in range(self.api_num)])

        with open(os.path.join(self.processed_info_dir,'statistics.csv'),'w') as f:
            f.write('mashup num,{}\n'.format(self.mashup_num))
            f.write('api num,{}\n'.format(self.api_num))
            f.write('mashup_description_lens,{}\n'.format(','.join(mashup_description_lens)))
            f.write('mashup_categories_lens,{}\n'.format(','.join(mashup_categories_lens)))
            f.write('api_description_lens,{}\n'.format(','.join(api_description_lens)))
            f.write('api_categories_lens,{}\n'.format(','.join(api_categories_lens)))
            f.write('api_provider_lens,{}\n'.format(','.join(api_provider_lens)))

        print('statistics,done!')

# service提供商的信息还没有录入，需改进
def get_mashup_api_field(id2info,id,field):
    """
    返回一个id的对应域的值
    :param id2info:
    :param id:
    :param field: final_des,各种类别   ['','']
    :return:['','']  当该id无信息或者无该域时，返回[]
    """
    info= id2info.get(id)
    if info is None or info.get(field) is None: # 短路
        return []
    else:
        return info.get(field)


def get_mashup_api_allCategories(mashup_or_api,id2info,id,Category_type='all'):
    """
    返回一个mashup api所有类别词
    :return:['','']  当该id无信息或者无该域时，返回[]
    """

    info= id2info.get(id)
    if mashup_or_api=='mashup' or new_Para.param.if_data_new: # 如果是新数据，mashup和api统一使用Categories
        Categories= get_mashup_api_field(id2info,id,'Categories')
        return Categories
    elif mashup_or_api=='api':
        Primary_Category=get_mashup_api_field(id2info,id,'Primary Category')
        Secondary_Categories= get_mashup_api_field(id2info,id,'Secondary Categories')
        if Category_type=='all':
            Categories=Primary_Category+Secondary_Categories
        elif Category_type=='first':
            Categories = Primary_Category
        elif Category_type=='second':
            Categories = Secondary_Categories
        return Categories
    else:
        raise ValueError('wrong mashup_or_api!')


def NLP_tool(str_): # 暂不处理
    return str_

# 没有安装NLTP时可以预先处理好再移植，之后不再调用NLTP
# english_stopwords = stopwords.words('english')  # 系统内置停用词
#
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# def NLP_tool(raw_description, SpellCheck=False):  # 功能需进一步确认！！！
#     """
#     返回每个文本预处理后的词列表:
#     return [[],[]...]
#     """
#
#     """ 拼写检查
#     d=None
#     if SpellCheck:
#         d = enchant.Dict("en_US")
#     """
#
#     # st = LancasterStemmer()  # 词干分析器
#
#     words = []
#     """
#     line = re.sub(punctuaion, ' ', text)  # 不去标点，标点有一定含义
#     words= line.split()
#     """
#     for sentence in tokenizer.tokenize(raw_description):  # 分句再分词
#         # for word in WordPunctTokenizer().tokenize(sentence): #分词更严格，eg:top-rated会分开
#         for word in word_tokenize(sentence):
#             word=word.lower()
#             if word not in english_stopwords and word not in domain_stopwords:  # 非停用词
#                 """
#                 if SpellCheck and not d.check(word):#拼写错误，使用第一个选择替换？还是直接代替？
#                     word=d.suggest(word.lower())[0]
#                 """
#                 # word = st.stem(word)   词干化，词干在预训练中不存在怎么办? 所以暂不使用
#                 words.append(word)
#
#     return words
#
#
# def test_NLP(text):
#     for sentence in tokenizer.tokenize(text):  # 分句再分词
#         for word in WordPunctTokenizer().tokenize(sentence):
#             print(word + "\n")

def test_utf():
    data_path = r'../mashup/%E2%96%B2hail'
    with open(data_path,encoding='utf-8') as f:
        print(f.readline())

if __name__ == '__main__':
    # test_NLP('i love you, New York.')

    test_data_dir = r'../test_data'
    real_data_dir= r'../data'
    pd = process_data(real_data_dir,True) #

    """
    for name, info in pd.get_mashup_api_info('mashup').items():
        print(name,info.get('final_description'))
    for name, info in pd.get_mashup_api_info('api').items():
        print(name, info.get('final_description'))
        print(name, info.get('Secondary Categories'))
    """

    """
    mashup_api_pairs = pd.get_mashup_api_pair('list')
    print(mashup_api_pairs)
    # 不存在信息的mashup/api的info获取
    name2info = pd.get_mashup_api_info('api')
    print(name2info)
    
    api_id2info=pd.get_mashup_api_id2info('api')
    for id, info in api_id2info.items():
        print(info.get('final_description'))
    """