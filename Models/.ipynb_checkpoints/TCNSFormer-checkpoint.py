import numpy as np
from torch import nn
import torch
from Models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Models.position_shapelet import PPSN
from Shapelet.auto_pisd import auto_piss_extractor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm
from Models.Transformer_EncDec import Encoder, EncoderLayer,ShapeAttention
import time

class TCNBlock(nn.Module):  #TCN的块，可以在TCN中重复使用
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout=0.1):
        super(TCNBlock, self).__init__()
        #一维卷积，
        #in_channels：输入的通道数，即每个输入样本中包含的特征数量
        #out_channels:输出的通道数，即有多少个卷积核，就有多少输出的通道，每一个通道是一列时序数据
        #kernel_size：卷积核的大小，***！!用同一个卷积核(eg:1*3)在每一个特征数据卷积后相加！!**
        #padding:两端填充的数据个数
        #dilation：感受野的间隔大小
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()
        self.padding=padding
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        #对输出的通道数(输出的特征数)归一化
        #对最后一维归一化
        self.norm=nn.LayerNorm(out_channels)
    def forward(self, x):
        
        #x是要输入的时序数据，self.conv1是对象属性，
        #使用 __call__ 方法可以将整个对象变为可调用函数，self.conv1持有卷积层的权重、偏置等参数，可以像函数一样被调用
        out = self.conv1(x)  #相当于out = self.conv1.__call__(x)
        #对输出的张量裁剪掉最后padding个时间步，contiguous为内存连续性处理
        out=out[:, :, :-self.padding].contiguous()
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out=out[: , :, :-self.padding].contiguous()
        out = self.relu(out)
        out = self.dropout(out)  
        #print(f'xxxx.shape:{x.shape}')
        #print(f'out.shape:{out.shape}')
        #batch_size*out_channels*sequence_length,三维数据，交换第二维和第三维的数据，nn.LayerNorm()默认对数据的最后一维归一化
        return self.norm((out + x).permute(0,2,1)).permute(0,2,1)

class TCN(nn.Module):
    def __init__(self, input_size, out_channels, num_blocks, kernel_size=3, dropout=0.1):
        #input_size：输入数据的特征维度，通常是输入的通道数
        #out_channels：输出通道数
        #num_blocks：TCN 的层数，也就是有多少个 TCNBlock
        #kernel_size：卷积核的大小
        #dropout：随机神经元权重失活的比例
        #查找TCN的父类(可能不止一个，依次实例化)，并调用父类的构造函数(__init__)，确保 TCN 继承了nn.Module的所有功能
        super(TCN, self).__init__()

        layers = []
        num_levels = num_blocks   
        for i in range(num_levels):

            # 膨胀因子的大小,2^i,i从0开始,2^0=1,2^1=2,2^2=4
            dilation_size = 2 ** i  

            #python的if语句
            in_channels = input_size if i == 0 else out_channels

            #两端填充的元素个数，保证输出序列的长度与输入序列相等
            padding = (kernel_size - 1) * dilation_size
            
            #实例化类 TCNBlock 对象添加到 layers 列表中，每一层有不同的膨胀率（dilation）和填充（padding）
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation_size, padding, dropout))
        
        #nn.Sequential 是一个用于组合顺序网络层的容器模块。它将每一层按顺序存储，其内部就是通过列表的形式存储模块，
        #  并依次调用它们的forward()方法，将输出作为输入传递给下一个模块
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_factory(config):
    if config['Net_Type'][0] == "PPSN":
        model = PPSN(shapelets_info=config['shapelets_info'], shapelets=config['shapelets'],
                                        len_ts=config['len_ts'], num_classes=config['num_labels'],
                                        sge=config['sge'], window_size=config['window_size'])
        config['shapelets'] = None
    elif config['Net_Type'][0] ==  "Shapeformer":
        model = Shapeformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] ==  "TCNSFormer":
        model = TCNSFormer(config)
    elif config['Net_Type'][0]==  "ST":
        model = TCNSFormer(config)
    else:
        raise ValueError(f"Unknown model_name: {config['Net_Type']}")
    return model

class ShapeBlock(nn.Module):
    def __init__(self, shapelet_info=None, shapelet=None, shape_embed_dim=32, window_size=50, len_ts=100, norm=1000, max_ci=3):
        super(ShapeBlock, self).__init__()
        self.dim = shapelet_info[5]
        self.shape_embed_dim = shape_embed_dim
        self.shapelet = torch.nn.Parameter(torch.tensor(shapelet), requires_grad=True)
        self.window_size = window_size
        self.norm = norm
        self.kernel_size = shapelet.shape[-1]
        self.weight = shapelet_info[3]

        self.ci_shapelet = np.sqrt(np.sum((shapelet[1:]- shapelet[:-1])**2)) + 1/norm
        self.max_ci = max_ci

        self.sp = shapelet_info[1]
        self.ep = shapelet_info[2]

        self.start_position = int(shapelet_info[1] - window_size)
        self.start_position = self.start_position if self.start_position >= 0 else 0
        self.end_position = int(shapelet_info[2] + window_size)
        self.end_position = self.end_position if self.end_position < len_ts else len_ts

        self.l1 = nn.Linear(self.kernel_size,shape_embed_dim)

    def forward(self, x):
        pis = x[:, self.dim, self.start_position:self.end_position]
        ci_pis = torch.square(torch.subtract(pis[:, 1:], pis[:, :-1]))

        pis = pis.unfold(1, self.kernel_size, 1).contiguous()
        pis = pis.view(-1, self.kernel_size)

        ci_pis = ci_pis.unfold(1, self.kernel_size - 1, 1).contiguous()
        ci_pis = ci_pis.view(-1, self.kernel_size - 1)
        ci_pis = torch.sum(ci_pis, dim=1) + (1 / self.norm)

        ci_shapelet_vec = torch.ones(ci_pis.size(0), device=x.device, requires_grad=False)*self.ci_shapelet
        max_ci = torch.max(ci_pis, ci_shapelet_vec)
        min_ci = torch.min(ci_pis, ci_shapelet_vec)
        ci_dist = max_ci / min_ci
        ci_dist[ci_dist > self.max_ci] = self.max_ci
        dist1 = torch.sum(torch.square(pis - self.shapelet), 1)
        dist1 = dist1 * ci_dist
        dist1 = dist1 / self.shapelet.size(-1)
        dist1 = dist1.view(x.size(0), -1)

        # soft-minimum
        index = torch.argmin(dist1, dim=1)
        pis = pis.view(x.size(0), -1, self.kernel_size)
        out = pis[torch.arange(int(x.size(0))).to(torch.long).cuda(), index.to(torch.long)]
        out = self.l1(out)

        return out.view(x.shape[0],1,-1)

class TCNSFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Shapelet Query  ---------------------------------------------------------
        num_classes=config['num_labels']
        num_heads = config['num_heads']
        self.shapelet_info = config['shapelets_info'] #子序列的信息
        self.shapelet_info = torch.IntTensor(self.shapelet_info)
        self.shapelets = config['shapelets']
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(config['dropout'])


        # Global Information
        self.shape_blocks = nn.ModuleList([
            ShapeBlock(shapelet_info=self.shapelet_info[i], shapelet=self.shapelets[i],
                       shape_embed_dim=config['shape_embed_dim'], len_ts=config["len_ts"])
            for i in range(len(self.shapelet_info))])
        self.shapelet_info = config['shapelets_info']
        print(f"shapelet_info:{len(self.shapelet_info)}")

        self.shapelet_info = torch.FloatTensor(self.shapelet_info)

        # Parameters Initialization -----------------------------------------------
        emb_size = config['shape_embed_dim']
        self.flatten = nn.Flatten()
        self.out = nn.Linear(128*(self.shapelet_info.shape[0]),128)
        self.out1 = nn.Linear(128, num_classes)
        self.TCN=TCN(len(self.shapelet_info),len(self.shapelet_info),config['num_blocks']) #BasicMotions


        self.linear1=nn.Linear(config['wordlen'],128) 
        print(f"config_wordlen:{config['wordlen']}")  #HandMovementDirection = 400 ,序列长度
        #print('暂停一下')
        #time.sleep(10000) # 暂停 10000秒
        # Merge Layer----------------------------------------------------------
        self.encoder = Encoder([EncoderLayer( ShapeAttention(emb_size, num_heads, config['dropout']), #注意力机制
                                             128,   # 输入/隐藏层维度
                                             256,   # 前馈网络维度
                                             dropout=config['dropout'], # Dropout 概率
                                            ) 
                                            for _ in range(config['num_layer'])  # 按 num_layer 生成多层 EncoderLayer
                               ],
                            norm_layer=nn.LayerNorm(128)  # 最后的 LayerNorm
                          )
        self.conv1 = nn.Conv1d(config['channel_num'], len(self.shapelet_info), padding=1,kernel_size=3) #channel_num为变量数
        print(f"channel_num:{config['channel_num']}")  #HandMovementDirection = 400 ,序列长度
        print(f"self.shapelet_info.shape[0]:{self.shapelet_info.shape[0]}")  
        
        
    
    def forward(self, x, ep):
        #print(f'x.shape:{x.shape}')

        attention_weights_all_layers = []
        h=self.dropout(self.activation(self.linear1(x))) #调整序列长度
        h=self.conv1(h) #调整通道数
        # print(f'h.shape:{h.shape}')
        tcn_res= self.TCN(h)
        #print(f'tcn_res.shape:{tcn_res.shape}')
        
        #tcn_res=self.conv1(x) #调整通道数
        #tcn_res= self.TCN(tcn_res)  
        #tcn_res=self.dropout(self.linear1(tcn_res)) #调整序列长度
        # Global information

        global_x = None
        for block in self.shape_blocks:
            if global_x is None:
                global_x = block(x)
            else:
                global_x = torch.cat((global_x, block(x)), dim=1)
                # print(global_x.shape)
        #print(f'global_x.shape:{global_x.shape}')
        #print('暂停一下')
        #time.sleep(10000) # 暂停 10000秒
        global_out,attention_weights_all_layers=self.encoder(global_x,tcn_res)
        #global_out=self.encoder(global_x,global_x)
        #global_out=self.encoder(tcn_res,tcn_res)
        
        # print(global_out.shape)
        out=self.dropout(self.activation(self.out(self.flatten(global_out))))
        out=self.dropout(self.out1(out))

        return out,attention_weights_all_layers


if __name__ == '__main__':
    print()



