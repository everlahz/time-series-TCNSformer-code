import os
import argparse
import logging
import time
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, dataset_class, Data_Verifier
from Models.TCNSFormer import model_factory, count_parameters
# from Models.shapeformer import model_factory, count_parameters
import multiprocessing
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner
from Shapelet.mul_shapelet_discovery import ShapeletDiscover
import torch
import random
# 设置随机种子


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger('__main__')   #获取一个名称为 __main__ 的日志记录器

parser = argparse.ArgumentParser()   #argparse是 Python 的标准库，用于解析命令行参数，创建一个ArgumentParser实例

# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/UEA/', choices={'Dataset/UEA/', 'Dataset/Segmentation/'},
                    help='Data path')  #--表示为可选参数，如果赋值，只能是choices里面的两种；默认为'Dataset/UEA/'。

parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
                    #作为保存结果的根目录

parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
                    #数据是否归一化，默认no

parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
                    #训练集中的验证比例,默认20%用于验证

parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
                   #10个批次打印一次批处理信息

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['ST'], choices={'T', 'C-T', 'PPSN', 'ST'}, help="Network Architecture. Convolution (C)"
                                                                              "Transformers (T)")
                    #可用的网络架构选项，包含卷积(C)和Transformers

# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=32, help='Internal dimension of transformer embeddings')
                   #Transformer 嵌入的内部维度,每个词在 Transformer 模型中的表示将是一个 32 维的向量，维度越大，拟合效果越好

parser.add_argument('--dim_ff', type=int, default=512, help='Dimension of dense feedforward part of transformer layer')
                   #前馈网络的维度，每一层的输出向量的大小
parser.add_argument('--num_heads', type=int, default=32, help='Number of multi-headed attention heads')
                   #多头注意力的头数（shapelet->transformer）
parser.add_argument('--local_num_heads', type=int, default=16, help='Number of multi-headed attention heads')
                   #本地多头注意力的头数(卷积->transformer)
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='Learn',
                    help='Fix Position Embedding')
                   #位置编码
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                    help='Relative Position Embedding')
                    #相对位置编码

# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
                   #进行训练的轮数,数据跑完一次
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
                   #每一轮分为多个批次，每一批次的大小是bath_size
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
                   #学习率
parser.add_argument('--weight_decay', type=float, default=5e-4, help='learning rate')
                   #权重衰减（L2正则化）系数
parser.add_argument('--dropout', type=float, default=0.4, help='Droupout regularization ratio')
                   #dropout 正则化的比率
parser.add_argument('--val_interval', type=int, default=1, help='Evaluate on validation every XX epochs. Must be >= 1')
                   #每 XX 个轮次后进行一次验证评估的频率
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                    help='Metric used for defining best epoch')
                   #最佳训练轮次的评估指标

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
                   #默认使用第一个GPU
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
                   #优化输出格式
parser.add_argument('--seed', default=1, type=int, help='Seed used for splitting sets')

# Local Information
parser.add_argument("--len_w", default=64, type=float, help="window size")
                   #窗口大小
parser.add_argument("--local_embed_dim", default=32, type=int, help="embedding dimension of shape")
                   #局部形状的嵌入维度
parser.add_argument("--local_pos_dim", default=32,type=int, help="embedding dimension of pos")
                   #位置的嵌入维度
#TCN
parser.add_argument("--num_blocks", default=3, type=int, help="number of TCN blocks")

#EnCoder-DeCoder
parser.add_argument("--num_layer", default=3, type=int, help="number of encoder layer")

# Global Information
parser.add_argument("--num_pip", default=0.2, type=float, help="number of pips")

parser.add_argument("--sge", default=0, type=int, help="stop-gradient epochs")
                   #停止梯度的轮次
parser.add_argument("--shape_embed_dim", default=128, type=int, help="embedding dimension of shape")
                   #形状嵌入的维度
parser.add_argument("--pos_embed_dim", default=128, type=int, help="embedding dimension of pos")
                   #位置嵌入的维度
parser.add_argument("--processes", default=64, type=int, help="number of processes for extracting shapelets")
                   #提取形状的进程数量

parser.add_argument("--pre_shapelet_discovery", default=1, type=int, help="number of processes for extracting shapelets")
                   #提取形状前的进程数量
parser.add_argument("--pre_extract_candidate", default=0, type=int, help="number of processes for extracting shapelets")
parser.add_argument("--dataset_pos", default=1, type=int, help="dataset position")
                   #使用第几个数据集
parser.add_argument("--window_size", default=100, type=int, help="window size")
                   #窗口大小
parser.add_argument("--num_shapelet", default=10, type=int, help="number of shapelets")

args = parser.parse_args() #解析命令行传入的参数，有不符合要求的参数会报错

#print(args)


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    multiprocessing.set_start_method("spawn")
    config = Setup(args)  # 将args参数对象创建一个配置字典
    print(args.num_shapelet)
    print(config['Net_Type'][0])
    #print('暂停一下')
    #time.sleep(10000) # 暂停 10000秒

    device = Initialization(config) #配置初始化计算设备（CPU 或 GPU），并设置随机种子
    Data_Verifier(config)  # Download the UEA and HAR datasets if they are not in the directory
    All_Results = ['Datasets', 'ConvTran']  # Use to store the accuracy of ConvTran in e.g "Result/Datasets/UEA"
    #初始化一个列表，All_Results 用来存储不同数据集及模型（如 ConvTran）的结果。
    
    list_dataset_name = os.listdir(config['data_path']) #返回一个包含数据集名称的列表
    list_dataset_name.sort()  #列表按照字母顺序排序
    print(list_dataset_name)  #打印出排序后的数据集名称列表

    problem = list_dataset_name[config['dataset_pos']]  # for loop on the all datasets in "data_dir" directory
    #获取 list_dataset_name 列表中特定索引（config['dataset_pos']）的项目。这应该表示当前要处理的数据集

    config['data_dir'] = config['data_path'] +"/"+ problem
    #更新配置字典config，将 data_dir 设置为数据集的具体目录
    print(f"config['data_dir']:{config['data_dir']}")
    
    print(f"problem:{problem}")
    #print(DataLoader.__module__)
    # ------------------------------------ Load Data ---------------------------------------------------------------
    logger.info("Loading Data ...") #日志记录
    # for i in config:
    #   print(i)
    Data = Data_Loader(config)      #导入数据集，把数据分批
    print(f"Data:{Data.keys()}")
    # for i in Data:
    #   print(i)
    
    train_data = Data['train_data']
    train_label = Data['train_label']
    len_ts = Data['max_len']  #时间序列的最大长度max_len
    dim = train_data.shape[1] #第二维张量  (batchsize,变量个数，序列长度)
    print(f"train_data_len:{len(train_data)}")
    print(f"train_data.shape:{train_data.shape}")
    print(f"test_data.shape:{Data['test_data'].shape}")
    #print(f"train_label:{train_label}")
    
    print("Number of shapelet:%s  window_size:%s" % (config['num_shapelet'], config['window_size']))
    #print('暂停一下')
    #time.sleep(10000) # 暂停 10000秒
    
    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Shapelet Discovery ----------------------------------------------
    shapelet_discovery = ShapeletDiscover(window_size=args.window_size, num_pip=args.num_pip,
                                          processes=args.processes, len_of_ts=len_ts, dim=dim)

    sc_path = f"store/{problem}_{args.window_size}.pkl"
    sc_path1 = f"store/extract_candidate{problem}_{args.window_size}.pkl"
    
    if args.pre_shapelet_discovery == 1:
        # 跳过形状基元提取和发现，直接从保存的文件中加载形状候选项
        print(f"Skipping shapelet extraction and discovery. Loading candidates from {sc_path}...")
        shapelet_discovery.load_shapelet_candidates(path=sc_path)
    else:
        if args.pre_extract_candidate == 1:
            shapelet_discovery.load_extract_candidate(sc_path1)
        else:
            shapelet_discovery.extract_candidate(train_data=train_data)
            shapelet_discovery.save_extract_candidate(sc_path1)
        # 检查是否已保存 extract_candidate 数据
        time_s = time.time()
        shapelet_discovery.discovery(train_data=train_data, train_labels=train_label)
        shapelet_discovery.save_shapelet_candidates(path=sc_path)
        print(f"Shapelet discovery time: {time.time() - time_s:.2f} seconds")

    #print('暂停一下')
    #time.sleep(10000) # 暂停 10000秒
    #获取指定数量的形状基元信息，返回一个二维数组，假设每一行代表一个形状基元的信息
    shapelets_info = shapelet_discovery.get_shapelet_info(number_of_shapelet=args.num_shapelet)

    print(f"shapelets_info.shape{shapelets_info.shape}") 
    
    #将 shapelets_info 的第四列转换为 PyTorch 张量 sw，
    sw = torch.tensor(shapelets_info[:,3])

    #Softmax 将放大的权重转换为一个归一化的概率分布，并扩展回原始权重的范围。
    #sw = torch.softmax(sw*20, dim=0)*sw.shape[0]
    sw = torch.softmax(sw.float()*20, dim=0)*sw.shape[0]
    #将 Softmax 处理后的权重从张量转换回 NumPy 数组，并重新赋值给 shapelets_info 的第四列，完成对形状基元信息中权重的更新
    shapelets_info[:,3] = sw.numpy()

    #print(shapelets_info)
    print(shapelets_info.dtype)
    shapelets = []
    for si in shapelets_info:
        print(si[0])
        print(si[5])
        print(si[1])
        print(si[2])
        sc = train_data[int(si[0]), int(si[5]), int(si[1]):int(si[2])]
        shapelets.append(sc)
        
    config['shapelets_info'] = shapelets_info  #self.shapelet_info.shape[0]表示序列长度
    config['shapelets'] = shapelets
    config['len_ts'] = len_ts
    config['ts_dim'] = dim

    # print(shapelets_info)
    #print('暂停一下')
    #time.sleep(10000) # 暂停 10000秒

    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'])
    val_dataset = dataset_class(Data['val_data'], Data['val_label'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------
    dic_position_results = [config['data_dir'].split('/')[-1]]

    #logger.info("Creating model ...")
    config['Data_shape'] = Data['train_data'].shape
    config['num_labels'] = int(max(Data['train_label']))+1
    
    config['channel_num']=next(enumerate(train_loader))[1][0].shape[1]
    config['wordlen']=next(enumerate(train_loader))[1][0].shape[2]
    
    model= model_factory(config) #实例化训练对象
    
    #logger.info("Model:\n{}".format(model))  打印整个网络的结构和参数
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    # -------------------------------------------- Model Initialization ------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'], problem + 'model_{}.pth'.format('last'))
    tensorboard_writer = SummaryWriter('summary')
    model.to(device)
    # ---------------------------------------------- Training The Model ------------------------------------
    logger.info('Starting training...')
    
    
    trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
                                print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
                                
    val_evaluator = SupervisedTrainer(model, val_loader, device, config['loss_module'],
                                      print_interval=config['print_interval'], console=config['console'],
                                      print_conf_mat=False)
    test_evaluator = SupervisedTrainer(model, test_loader, device, config['loss_module'],
                                      print_interval=config['print_interval'], console=config['console'],
                                      print_conf_mat=False)

    train_runner(config, model, trainer, val_evaluator,test_evaluator,save_path)
    best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_model.to(device)

    best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, config['loss_module'],
                                            print_interval=config['print_interval'], console=config['console'],
                                            print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print(print_str)
    dic_position_results.append(all_metrics['total_accuracy'])
    problem_df = pd.DataFrame(dic_position_results)
    problem_df.to_csv(os.path.join(config['pred_dir'] + '/' + problem + '.csv'))

    All_Results = np.vstack((All_Results, dic_position_results))

    All_Results_df = pd.DataFrame(All_Results)
    All_Results_df.to_csv(os.path.join(config['output_dir'], 'ConvTran_Results.csv'))

