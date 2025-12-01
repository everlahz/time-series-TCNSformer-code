import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pandas as pd

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class ShapeAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
  
        self.d_k = emb_size // num_heads 
        self.d_model = emb_size
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)
        self.attn = None

    def forward(self, q,x):
        #batch_size, seq_len, _ = x.shape
        # 线性投影并分割为多个头
        # k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # q = self.query(q).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)
        batch_size = q.size(0)
        query = self.query(q)  
        q=q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k= self.key(x).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 3, 1)
        v= self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        
        #print(f"q后的shape:{q.shape}")
        #print(f"k后的shape:{k.shape}")
        attn = torch.matmul(q, k) * self.scale
       # print(attn.shape)
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)
        
        # import matplotlib.pyplot as plt
        # plt.plot(x[0, :, 0].detach().cpu().numpy())
        # plt.show()
        self.attn = attn

        out = torch.matmul(attn, v)  #矩阵乘法
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        #out = out.reshape(batch_size, seq_len, -1)
       # print(f'out.shape:{out.shape}')
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out,attn

    def get_att(self):
        return self.attn
        
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Linear(d_model,d_ff)
        self.conv2 = nn.Linear(d_ff,d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q,x, attn_mask=None, tau=None, delta=None):
        new_x,attention_weights = self.attention(q,x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return self.norm2(x + y),attention_weights


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, q ,x):
        # x [B, L, D]
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x = attn_layer(q,x)
                x = conv_layer(x)
            x= self.attn_layers[-1](q,x)
        else:
            attention_weights_all_layers = []
            for attn_layer in self.attn_layers:
                x,attention_weights= attn_layer(q,x)
                attention_weights_all_layers.append(attention_weights)

        if self.norm is not None:
            x = self.norm(x)

        return x,attention_weights_all_layers

