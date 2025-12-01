import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def hotmap(attention_weights_all_layers,save_path="heatmap.png"):
    attention_weights_last_layer = attention_weights_all_layers[-1][0, 0].detach().cpu().numpy()
    #print(f'attention_weights_last_layer:{attention_weights_last_layer}')
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights_last_layer,
        cmap="RdBu_r",            # 深蓝到深红渐变配色
        xticklabels=200,            # 每隔 2 个显示一个列标签
        yticklabels=200,            # 每隔 2 个显示一个行标签
        cbar_kws={"label": "Attention Weight"}  # 添加颜色条
    )
    plt.title("Self-Attention Heatmap (Last Layer, Head 1)", fontsize=14)
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 高分辨率保存
    plt.close()  # 关闭当前绘图窗口，释放内存