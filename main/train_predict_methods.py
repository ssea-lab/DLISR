import sys

from recommend_models.CI_Model import CI_Model
from recommend_models.NI_Model_online_new import NI_Model_online

sys.path.append("..")
# from recommend_models.text_tag_continue_model import gx_text_tag_continue_only_MLP_model
from main.dataset import dataset
from main.new_para_setting import new_Para

from main.helper import get_iniFeaturesAndParas, load_trained_model

import os
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import utils
from main.evalute import evalute, evalute_by_epoch, summary
import matplotlib.pyplot as plt

def train_save_best_NDCG_model(recommend_model, model,train_data,test_data,start_epoch=0,true_candidates_dict=None,CI_start_test_epoch=2,earlyStop_epochs=5):
    """
    训练多个epoch，每个之后均测试，选择并返回NDCG等最终指标最优的模型
    :param recommend_model:  整体的推荐模型
    :param model:  model_core
    :param train_data:
    :param test_data:
    :param start_epoch: 之前该模型已经训练过多个epoch，在这个基础上接着训练
    :param true_candidates_dict: ？
    :return:
    """
    print('training_save_best_NDCG_model...')
    epoch_evaluate_results = []

    train_labels = train_data[-1]
    if new_Para.param.final_activation == 'softmax': # 针对softmax变换labels
        train_labels = utils.to_categorical(train_labels,num_classes=2)

    # ??? 跟获取的实例有关
    test_phase_flag = False if (new_Para.param.pairwise or new_Para.param.NI_OL_mode =='OL_GE' or new_Para.param.train_mashup_best) else True # 在线NI时，也要区分训练和测试过程得到隐式表示的方法
    train_instances_tuple = recommend_model.get_instances(*train_data[:-1], test_phase_flag=test_phase_flag) # 有必要才获取负例:pairwise的训练

    # 读取之前训练过的最优指标
    if start_epoch>0:
        with open(dataset.crt_ds.new_best_epoch_path.format(recommend_model.get_simple_name()), 'r') as f:
            best_epoch = int(f.read().strip())
        with open(dataset.crt_ds.new_best_NDCG_path.format(recommend_model.get_simple_name()), 'r') as f:
            best_NDCG_5 = float(f.read().strip())
    else:
        best_epoch,best_NDCG_5 = 0,0

    # 有必要转化为pairwise模型
    train_model=recommend_model.get_pairwise_model() if new_Para.param.pairwise else model

    for epoch in range(new_Para.param.num_epochs-start_epoch):
        if start_epoch==0 and epoch==0: # 首次训练要编译
            if new_Para.param.pairwise:
                train_model.compile(optimizer=recommend_model.optimizer, loss=lambda y_true,y_pred: y_pred,metrics=['accuracy'])
            else:
                train_model.compile(optimizer=recommend_model.optimizer, loss='binary_crossentropy',metrics=['accuracy'])
            print('model compile,done!')

        if start_epoch>0: # 载入原模型，直接在原来的基础上训练
            train_model=load_trained_model(recommend_model, model)

        epoch = epoch+start_epoch
        print('Epoch {}'.format(epoch))

        # test_model = model if not new_Para.param.pairwise else recommend_model.get_single_model()  # pairwise时需要复用相关参数!!!

        if type(train_instances_tuple) ==  tuple:
            hist = train_model.fit([*train_instances_tuple], np.array(train_labels), batch_size=new_Para.param.batch_size,epochs=1, verbose=2,shuffle=True,validation_split=new_Para.param.validation_split)  #可以观察过拟合欠拟合 ,validation_split=0.1
        else:
            hist = train_model.fit(train_instances_tuple, np.array(train_labels),
                                   batch_size=new_Para.param.batch_size, epochs=1, verbose=2, shuffle=True,
                                   validation_split=new_Para.param.validation_split)
        print('model train,done!')

        # 记录：数据集情况，模型新旧完整，模型架构，训练设置
        # recommend_model.get_simple_name()+ '---'+
        model_name = dataset.crt_ds.data_name+ recommend_model.get_name() +new_Para.param.train_name if epoch == 0 else '' # 记录在测试集的效果，写入evalute.csv

        # 每个epoch的测试
        save_loss_acc(hist,model_name,epoch=epoch)

        if not os.path.exists(recommend_model.model_dir):
            os.makedirs(recommend_model.model_dir)

        if isinstance(recommend_model,CI_Model) and not isinstance(recommend_model,NI_Model_online):
            first_test_epoch = CI_start_test_epoch # 前3轮效果差，一般不用测
        else:
            first_test_epoch = 0

        if epoch<first_test_epoch: # 暂不测试，提高速度
            epoch_evaluate_results.append(None)
            continue

        if epoch == first_test_epoch: # 记录第一个epoch的测试时间
            with open(new_Para.param.time_path, 'a+') as f1:
                f1.write(recommend_model.get_simple_name())
                f1.write('\n')

        # test_model = model if not new_Para.param.pairwise else recommend_model.get_single_model()  # pairwise时需要复用相关参数!!!
        # 没必要使用get_model再获取，传入的model是对象引用，pairwise更新后model也变化

        # 每个epoch的测试
        epoch_evaluate_result = evalute_by_epoch(recommend_model, model, model_name,test_data, record_time=True if epoch==1 else False,true_candidates_dict = true_candidates_dict)
        epoch_evaluate_results.append(epoch_evaluate_result)

        if epoch_evaluate_result[0][3] >= best_NDCG_5:  # 优于目前的best_NDCG_5才存储
            best_NDCG_5 = epoch_evaluate_result[0][3]
            best_epoch = epoch
            model.save_weights(dataset.crt_ds.new_model_para_path.format(recommend_model.model_dir, epoch))  # 记录该epoch下的模型参数***
        else:
            if epoch - best_epoch >= earlyStop_epochs: # 大于若干个epoch，效果没有提升，即时终止
                break
        #@@@# # 第一个 epoch之后存储HIN_sim对象？？？ 删去only_MLP_model的判断，换成了CI? NI为什么要记录？or isinstance(recommend_model,NI_Model)
        # if epoch==0 and (isinstance(recommend_model,gx_text_tag_continue_only_MLP_model) ):
        #     recommend_model.save_HIN_sim()

        # 看word embedding矩阵是否发生改变，尤其是padding的0
        # print('some embedding parameters after {} epoch:'.format(epoch))
        # print (recommend_model.embedding_layer.get_weights ()[0][:2])

    # 记录最优epoch和最优NDCG@5
    with open(dataset.crt_ds.new_best_epoch_path.format(recommend_model.model_dir), 'w') as f:
        f.write(str(best_epoch))
    with open(dataset.crt_ds.new_best_NDCG_path.format(recommend_model.model_dir), 'w') as f:
        f.write(str(best_NDCG_5))
    print('best epoch:{},best NDCG@5:{}'.format(best_epoch,best_NDCG_5))

    # 记录最优指标
    csv_table_name ='best_indicaters\n'  # 命名格式！！！
    summary(new_Para.param.evaluate_path, csv_table_name, epoch_evaluate_results[best_epoch], new_Para.param.topKs)

    # 把记录的非最优的epoch模型参数都删除
    try:
        for i in range(new_Para.param.num_epochs):
            temp_path = dataset.crt_ds.new_model_para_path.format(recommend_model.model_dir, i)
            if i!=best_epoch and os.path.exists(temp_path):
                os.remove(temp_path)

        model.load_weights(dataset.crt_ds.new_model_para_path.format(recommend_model.model_dir, best_epoch))
    finally:
        return model


def save_loss_acc(train_log,model_name,epoch=0,if_multi_epoch=False):
    # if_multi_epoch：每次存一个epoch
    # 每个epoch存储loss,val_loss,acc,val_acc
    if not if_multi_epoch:
        with open(new_Para.param.loss_path,'a+') as f:
            if epoch==0: # 第一个epoch记录模型名
                f.write(model_name+'\n')
                if new_Para.param.validation_split==0:
                    f.write('epoch,loss,acc\n')
                else:
                    f.write('epoch,loss,val_loss,acc,val_acc\n')
            if new_Para.param.validation_split == 0:
                f.write('{},{},{}\n'.format(epoch, train_log.history["loss"][0],train_log.history["acc"][0]))
            else:
                f.write('{},{},{},{},{}\n'.format(epoch,train_log.history["loss"][0],train_log.history["val_loss"][0],train_log.history["acc"][0],train_log.history["val_acc"][0]))
    else:
        with open(new_Para.param.loss_path, 'a+') as f:
            f.write(model_name +'EarlyStop'+ '\n')
            if new_Para.param.validation_split == 0:
                f.write('epoch,loss,acc\n')
            else:
                f.write('epoch,loss,val_loss,acc,val_acc\n')
            epoch_num = len(train_log.history["loss"])
            for i in range(epoch_num):
                if new_Para.param.validation_split == 0:
                    f.write('{},{},{}\n'.format(i, train_log.history["loss"][i], train_log.history["acc"][i]))
                else:
                    f.write('{},{},{},{},{}\n'.format(i, train_log.history["loss"][i], train_log.history["val_loss"][i],
                                                  train_log.history["acc"][i], train_log.history["val_acc"][i]))


def train_monitoring_loss_acc_model(recommend_model, model, train_data):
    """
    绘制loss_acc曲线, 观察过拟合欠拟合
    """
    train_labels = train_data[-1]
    train_instances_tuple = recommend_model.get_instances(*train_data[:-1])
    model.compile(optimizer=Adam(lr=new_Para.param.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit([*train_instances_tuple], np.array(train_labels), batch_size=new_Para.param.small_batch_size, epochs=new_Para.param.num_epochs,
                         verbose=1, shuffle=True, validation_split=0.1)  # 可以观察过拟合欠拟合
    plot_loss_acc(hist, recommend_model.get_simple_name())
    return model

def plot_loss_acc(train_log,model_name):
    # 传入log对象，绘制曲线
    epochs= new_Para.param.num_epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on the model")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("Loss_Accuracy_{}.jpg".format(model_name))


def train_save_by_early_stop(recommend_model, model,train_data,test_data):
    """
    训练时按照验证集的loss，early stopping得到最优的模型；最后基于该模型测试
    :return:
    """
    if_Train = True if new_Para.param.pairwise else False
    train_labels = train_data[-1]
    train_instances_tuple = recommend_model.get_instances(*train_data[:-1], test_phase_flag=if_Train)

    train_model = recommend_model.get_pairwise_model() if new_Para.param.pairwise else model
    if new_Para.param.pairwise:
        train_model.compile(optimizer=recommend_model.optimizer, loss=lambda y_true, y_pred: y_pred, metrics=['accuracy'])
    else:
        train_model.compile(optimizer=recommend_model.optimizer, loss='binary_crossentropy',metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='min')
    hist = train_model.fit([*train_instances_tuple],train_labels, epochs=new_Para.param.num_epochs, batch_size=new_Para.param.small_batch_size, callbacks=[early_stopping],validation_split=new_Para.param.validation_split,shuffle=True) #
    model.save_weights(dataset.crt_ds.new_model_para_path.format(recommend_model.model_dir, 'min_loss')) # !!! 改正

    model_name = recommend_model.get_simple_name() + recommend_model.get_name() + '_min_loss'
    save_loss_acc(hist, model_name,if_multi_epoch=True)

    epoch_evaluate_result = evalute_by_epoch(recommend_model, model, model_name, test_data)
    return model

def load_preTrained_model(recommend_model, model, train_data, test_data,train_mode,train_new,true_candidates_dict=None):
    """
    各种模型(完全冷启动和部分冷启动，完整和部分的)都可以通用
    :param recommend_model:
    :param model:
    :param train_data: 与参数对应，是否加入slt_api_ids
    :param test_data:
    :param train_mode： 'best_NDCG' or 'min_loss'
    :param train_new： 是否重新训练模型
    :return:
    """
    # 模型相关的东西都放在该数据下的文件夹下,不同模型不同文件夹！！！

    model_dir = recommend_model.model_dir  #应该设为model的属性！！！
    if not os.path.exists(model_dir):
        print('makedirs for:',model_dir)
        os.makedirs(model_dir)

    if os.path.exists(dataset.crt_ds.new_best_epoch_path.format(model_dir)) and not train_new:  # 利用求过的结果，路径还需要改！！！
        print('preTrained model, exists!')
        return load_trained_model(recommend_model,model)
    else:
        if train_mode=='best_NDCG':
            model=train_save_best_NDCG_model(recommend_model, model, train_data, test_data,true_candidates_dict = true_candidates_dict)
        elif train_mode=='min_loss':
            model=train_save_by_early_stop(recommend_model, model, train_data, test_data)
        elif train_mode=='monitor loss&acc':
            train_monitoring_loss_acc_model(recommend_model, model, train_data, test_data)
        else:
            print('wrong train_mode:')
            print(train_mode)
        return model