# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   07.04.22
@Author  :   liu haoyu
@Version :   1.0
@Contact :   gibbsliuhy@gmail.com
@License :   (C)Copyright 2020-2021, 3dvc-NLPR-CASIA
@Desc    :   global 用sg之前的特征maxpooling 然后cat sg之前的local的
'''

# here to place import libs
import se_math.se3 as se3
from matplotlib import projections
import numpy as np
from numpy.random.mtrand import shuffle
import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
import os
import torch.nn.functional as F 
import torch.nn as nn
from torch.optim import optimizer

import torchvision
import argparse
import datetime
import time

import pct
import dataset
import sys

# se math
import se_math.se3 as se3
import se_math.invmat as invmat

# utils
import pointnet_util as pu

#  pl.seed_everything(42)

# import different models
#  import model_ae as ma
import pointtransformer_partseg as pp
#  import model_sort as ms

from PyTorchEMD.emd import earth_mover_distance

import metrics
import open3d as o3d

import matplotlib as mpb
mpb.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


import pdb

"""
#=============================================
fundamation
#=============================================
"""
def scaled_dot_production(q, k, v, mask=None):
    dk = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2,-1))
    attn_logits = attn_logits / math.sqrt(dk)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask ==0, -9e-15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


"""
#=============================================
Encoder
#=============================================
"""
class layerAttention(nn.Module):
    def __init__(self, config, embed_dim) -> None:
        super(layerAttention, self).__init__()
        self.C = config
        self.mlpq = nn.Linear(embed_dim, embed_dim//4)
        self.mlpk = nn.Linear(embed_dim, embed_dim//4)
        self.mlpv = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, xyz):
        q = self.mlpq(xyz)
        k = self.mlpk(xyz)
        v = self.mlpv(xyz)
        r = scaled_dot_production(q,k,v)
        attention = r[1]
        r = r[0]
        r = xyz - r
        r = xyz + F.relu(self.out(r))
        return r, attention


class Encoder(nn.Module):
    def __init__(self, config, num_points=1024) -> None:
        super(PCTransformer_nonsort, self).__init__()
        self.C = config
        feature_size = 64
        gs2_feature_size = 128
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64,feature_size)
        self.mlp3 = nn.Linear(feature_size+3, 128)
        self.mlp4 = nn.Linear(128, gs2_feature_size)
        self.mlp5 = nn.Linear(gs2_feature_size+3, gs2_feature_size*2)
        self.mlp6 = nn.Linear(gs2_feature_size*2, gs2_feature_size*2)
        
        self.bn1 = nn.BatchNorm1d(num_points)
        self.bn2 = nn.BatchNorm1d(num_points)

        self.sg1 = pu.sample_and_group
        self.fps = pu.farthest_point_sample
        self.sg2 = pu.sample_and_group
        
        #  self.atten1 = AttentionBlock(self.C, gs2_feature_size * 2 , gs2_feature_size * 2 , 4, 0.1)
        #  self.atten2 = AttentionBlock(self.C, gs2_feature_size * 2 , gs2_feature_size * 2 , 4, 0.1)
        #  self.atten3 = AttentionBlock(self.C, gs2_feature_size * 2 , gs2_feature_size * 2 , 4, 0.1)
        #  self.atten4 = AttentionBlock(self.C, gs2_feature_size * 2 , gs2_feature_size * 2 , 4, 0.1)
        self.atten1 = layerAttention(self.C, gs2_feature_size * 2)
        self.atten2 = layerAttention(self.C, gs2_feature_size * 2)
        self.atten3 = layerAttention(self.C, gs2_feature_size * 2)
        self.atten4 = layerAttention(self.C, gs2_feature_size * 2)


        self.out = nn.Linear(gs2_feature_size * 2 * 5, 1024)

    def forward(self, xyz):
        # print(xyz.shape)
        x = F.relu(self.bn1(self.mlp1(xyz)))
        x = F.relu(self.bn2(self.mlp2(x)))
        x, f1 = self.sg1(512, 0, 32, xyz, x, False, True) # [B, N , 3] [B, N, 32, 64+3]
        # print(x.shape)
        # print(f1.shape)
        f1f = F.relu(self.mlp3(f1))
        f1f = F.relu(self.mlp4(f1f))
        f1f = torch.max(f1f, dim=-2)[0]
        # print('f1f.shape', f1f.shape)
        x2, f2 = self.sg2(256, 0, 32, x, f1f, False, True) # [B, N, 3], [B, N, 32, 128+3] N = 256
        # print('aa', x2.shape)
        # print('aaa', f2.shape)
        f2f = F.relu(self.mlp5(f2))
        f2f = F.relu(self.mlp6(f2f))  
        f2f = torch.max(f2f, dim=-2)[0] # [B, 256, 256]
        #  att1 = self.atten1(f2f.transpose(0,1)) # TODO: 这里按照MultiHeadAttention的文档，将B和L颠倒
        att1, attention1 = self.atten1(f2f)  # 没有颠倒
        att2, attention2 = self.atten2(att1)
        att3, attention3 = self.atten3(att2)
        att4, attention4 = self.atten4(att3)
        att = torch.cat([att1, att2, att3, att4], dim=-1)
        attention = attention1 + attention2 + attention3 + attention4
        attention = attention / 4 # 这是对x2这些点的点特征，一对多，每个点都对应了其他256个点的特征
        #  print('f2faaaaa', f2f.shape)
        #  print('att,aaa', att.shape)
        att = torch.cat([att, f2f], dim=-1)
        # print('att', att.shape)
        out = self.out(att)
        out = torch.max(out, dim=1)[0]  # [B, 1024] 1024是特征长度

        return out, x2, attention
"""
#=============================================
Decoder
#=============================================
"""
class de_Attention_block(nn.Module):
    def __init__(self, config, n_head=4) -> None:
        super(de_Attention_block, self).__init__()
        self.C = config
        self.in_dim = 1024

        self.ln1 = nn.LayerNorm(1024)
        self.ln2= nn.LayerNorm(256)
        self.q_mlp = nn.Linear(256, self.in_dim//4)
        self.k_mlp = nn.Linear(self.in_dim, self.in_dim//4)
        self.v_mlp = nn.Linear(self.in_dim, self.in_dim//4)

        #  self.atten = nn.MultiheadAttention(self.in_dim//4, n_head)
        self.mlpx = nn.Linear(256, 256)

    def forward(self, xyz, pc):
        """
        @param xyz: [B, 16, 1024] 特征, Encoder_tensor
        @param pc: [B, 16, 256], Decoder_tensor after mlp1 in decoder
        """
        xyz = self.ln1(xyz)
        pc = self.ln2(pc) # B,16，256
        q = self.q_mlp(pc)
        k = self.k_mlp(xyz)
        v = self.v_mlp(xyz)
        #  x = pc - self.atten(q,k,v)[0]
        x = pc - scaled_dot_production(q,k,v)[0]
        x = self.mlpx(x)
        x = pc + x
        return x


class copy_and_mapping(nn.Module):
    def __init__(self, config, nmul, feature_size) -> None:
        super(copy_and_mapping, self).__init__()
        self.C = config
        self.fsize = feature_size // nmul # channel 320
        self.nmul = nmul

        # conv2d t
        self.cov2dt1 = nn.ConvTranspose2d(1280, 160, (1, 32), (1, 32))
        
        self.mlp0 = nn.Linear(1280, 512)
        self.mlp00 = nn.Linear(512, 160)
        self.mlp1 = nn.Linear(feature_size, 160)


    def forward(self, x):
        x = x.unsqueeze(3) # B, 1280, 4, 1
        #  print('asdfa sdf ', x.shape)
        x1 = self.cov2dt1(x) # B, 160, 4, 32 
        #  print('x1', x1.shape)
        #  print('x1', x1.shape)
        x2 = self.mlp1(x.permute(0,2,3,1)) # B, 4, 1, 160
        x2 = x2.repeat(1, 1, self.nmul, 1) # B, 4, 32, 160
        #  print('x1', x1.shape)
        #  print('x2', x2.shape)
        x = x1.permute(0,2,3,1) + x2
        x = x.flatten(1,2)
        return x # B, 128, 160


class copy_and_mapping2(nn.Module):
    def __init__(self, config, nmul, feature_size) -> None:
        super(copy_and_mapping2, self).__init__()
        self.C = config
        self.fsize = feature_size // nmul # channel 320
        self.nmul = nmul

        # conv2d t
        #  self.cov2dt1 = nn.ConvTranspose2d(1280, 160, (1, 32), (1, 32))
        self.conv2d = nn.Conv2d(1280, 160, (2,1), (2,2))
        
        self.mlp0 = nn.Linear(1280, 512)
        self.mlp00 = nn.Linear(512, 160)
        self.mlp1 = nn.Linear(feature_size, 160)


    def forward(self, x):
        x = x.unsqueeze(3) # B, 1280, 256, 1
        #  print('asdfa sdf ', x.shape)
        x1 = self.conv2d(x) # B, 160, 128, 1
        #  print('x1', x1.shape)
        #  print('x1', x1.shape)
        #  x2 = self.mlp1(x.permute(0,2,3,1)) # B, 4, 1, 160
        #  x2 = x2.repeat(1, 1, self.nmul, 1) # B, 4, 32, 160
        #  print('x1', x1.shape)
        #  print('x2', x2.shape)
        #  x = x1.permute(0,2,3,1) + x2
        #  x = x.flatten(1,2)
        x1 = x1.squeeze(-1) # B, 160, 128
        x = x1.permute(0, 2, 1)
        return x # B, 128, 160


class PCTransformerDecoder2(nn.Module):
    def __init__(self, config, num_points=1024) -> None:
        super(PCTransformerDecoder2, self).__init__()
        self.C = config
        
        self.mlp1 = nn.Linear(num_points+3, num_points//4)
        self.mlp2 = nn.Linear(num_points, num_points//4)

        self.att1 = de_Attention_block(self.C)
        self.att2 = de_Attention_block(self.C)
        self.att3 = de_Attention_block(self.C)
        self.att4 = de_Attention_block(self.C)

        # copy and mapping
        self.cm = copy_and_mapping2(config, 32, 1280)

        self.out1 = nn.Linear(160, 64)
        self.out2 = nn.Linear(64, 64)
        self.out3 = nn.Linear(64, 3)


    def forward(self, xyz, pointxyz):
        """
        @param xyz: [B, 1, feature_size]
        """
        #  print(xyz.shape)
        xyz = xyz.unsqueeze(1)
        x = xyz.repeat(1, 256, 1)
        seed = torch.eye(1, device=xyz.device)
        pc = x+seed
        pc = torch.cat([pointxyz, pc], dim=-1)
        pc = self.mlp1(pc)
        #  xyz_fe = self.mlp2(xyz)
        # pc -> de_tensor
        # x -> en_tensor
        x1 = self.att1(x, pc)
        x2 = self.att2(x, x1)
        x3 = self.att3(x, x2)
        x4 = self.att4(x, x3)
        x0 = torch.cat([x1, x2, x3, x4], dim=2) # [B, 256, 1024]
        x = torch.cat([x0, pc], dim=2) # [B, 256, 1280]
        #  print(x.shape)
        #  print(x0.shape)
        #  print(x1.shape)
        #  print(x2.shape)
        #  print(x3.shape)
        #  print(x4.shape)
        #  print('xxxxx', x.shape)
        x = self.cm(x.permute(0, 2, 1)) # [B, 128, 160 ]
        #  print('x', x.shape)
        x = F.relu(self.out1(x))
        x = F.relu(self.out2(x))
        x = self.out3(x)
        #  print(x.shape)
        return x # [B, 128, 3]

class BiDecoderNoneCross(nn.Module):
    def __init__(self, config, num_points=1024) -> None:
        super(BiDecoderNoneCross, self).__init__()
        self.C = config

        self.mlp1 = nn.Linear(512, 512)
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, 2)



    def forward(self, f_local, f_global):
        # f_local [B, 256, 1024]
        # f_global [B, 1, 1024]
        #  print(f_local.shape)
        #  print(f_global.shape)
        f_global = f_global.unsqueeze(1) if len(f_global.shape) ==2 else f_global #[B, 1, 1024]
        f = f_global.repeat(1,256,1) # [ B, 256, 1024 ]
        f = torch.cat([f_local, f], dim=1).permute(0,2,1) #[B,1024,512]
        #  print(f.shape)
        #  f = self.mlp1()
        #  f = F.sigmoid(self.mlp1(f))
        #  f = F.sigmoid(self.mlp2(f))
        f = F.relu(self.mlp1(f))
        f = F.relu(self.mlp2(f))
        f = self.mlp3(f)
        #  pdb.set_trace()
        return f.permute(0,2,1) # [B, 2, 1024]


class PCTransformerDecoder(nn.Module):
    ##
    # @brief 用的就是这个
    def __init__(self, config, num_points=1024) -> None:
        super(PCTransformerDecoder, self).__init__()
        self.C = config
        
        self.mlp1 = nn.Linear(num_points, num_points//4)
        self.mlp2 = nn.Linear(num_points//4, num_points//4)

        self.att1 = de_Attention_block(self.C)
        self.att2 = de_Attention_block(self.C)
        self.att3 = de_Attention_block(self.C)
        self.att4 = de_Attention_block(self.C)

        # copy and mapping
        self.cm = copy_and_mapping(config, 32, 1280)

        self.out1 = nn.Linear(160, 64)
        self.out2 = nn.Linear(64, 64)
        self.out3 = nn.Linear(64, 3)


    def forward(self, xyz):
        """
        @param xyz: [B, 1, feature_size]
        """
        #  print(xyz.shape)
        xyz = xyz.unsqueeze(1)
        x = xyz.repeat(1, 4, 1)
        seed = torch.eye(1, device=xyz.device)
        pc = x+seed
        pc = self.mlp1(pc)  # B, 4, 256
        # pc -> de_tensor
        # x -> en_tensor
        x1 = self.att1(x, pc)
        x2 = self.att2(x, x1)
        x3 = self.att3(x, x2)
        x4 = self.att4(x, x3)
        x0 = torch.cat([x1, x2, x3, x4], dim=2) # [B, 4, 1024]
        x = torch.cat([x0, pc], dim=2) # [B, 4, 1280]
        #  print(x.shape)
        #  print(x0.shape)
        #  print(x1.shape)
        #  print(x2.shape)
        #  print(x3.shape)
        #  print(x4.shape)
        #  print('xxxxx', x.shape)
        x = self.cm(x.permute(0, 2, 1)) # [B, 128, 160 ]
        #  print('x', x.shape)
        x = F.relu(self.out1(x))
        x = F.relu(self.out2(x))
        x = self.out3(x)
        #  print(x.shape)
        return x # [B, 128, 3]

class PCTransformer_nonsort(nn.Module):
    def __init__(self, config, num_points=1024) -> None:
        super(PCTransformer_nonsort, self).__init__()
        self.C = config
        feature_size = 64
        gs2_feature_size = 128
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64,feature_size)
        self.mlp3 = nn.Linear(feature_size+3, 128)
        self.mlp4 = nn.Linear(128, gs2_feature_size)
        self.mlp5 = nn.Linear(gs2_feature_size+3, gs2_feature_size*2)
        self.mlp6 = nn.Linear(gs2_feature_size*2, gs2_feature_size*2)
        
        self.bn1 = nn.BatchNorm1d(num_points)
        self.bn2 = nn.BatchNorm1d(num_points)

        self.sg1 = pu.sample_and_group
        self.fps = pu.farthest_point_sample
        self.sg2 = pu.sample_and_group
        
        #  self.atten1 = AttentionBlock(self.C, gs2_feature_size * 2 , gs2_feature_size * 2 , 4, 0.1)
        #  self.atten2 = AttentionBlock(self.C, gs2_feature_size * 2 , gs2_feature_size * 2 , 4, 0.1)
        #  self.atten3 = AttentionBlock(self.C, gs2_feature_size * 2 , gs2_feature_size * 2 , 4, 0.1)
        #  self.atten4 = AttentionBlock(self.C, gs2_feature_size * 2 , gs2_feature_size * 2 , 4, 0.1)
        self.atten1 = layerAttention(self.C, gs2_feature_size * 2)
        self.atten2 = layerAttention(self.C, gs2_feature_size * 2)
        self.atten3 = layerAttention(self.C, gs2_feature_size * 2)
        self.atten4 = layerAttention(self.C, gs2_feature_size * 2)


        self.out = nn.Linear(gs2_feature_size * 2 * 5, 1024)

    def forward(self, xyz):
        #  print(xyz.shape)
        #  if len(xyz.shape) == 2 and xyz.shape[0] == 1024:
            #  xyz = xyz.unsqueeze(0)
        x_feature = F.relu(self.bn1(self.mlp1(xyz)))
        x_feature = F.relu(self.bn2(self.mlp2(x_feature))) # [B, N, 64]
        x, f1 = self.sg1(512, 0, 32, xyz, x_feature, False, True) # [B, N , 3] [B, N, 32, 64+3]
        # print(x.shape)
        # print(f1.shape)
        f1f = F.relu(self.mlp3(f1))
        f1f = F.relu(self.mlp4(f1f))
        f1f = torch.max(f1f, dim=-2)[0] # 特征融合
        # print('f1f.shape', f1f.shape)
        x2, f2 = self.sg2(256, 0, 32, x, f1f, False, True) # [B, N, 3], [B, N, 32, 128+3] N = 256
        # print('aa', x2.shape)
        # print('aaa', f2.shape)
        f2f = F.relu(self.mlp5(f2))
        f2f = F.relu(self.mlp6(f2f))  
        f2f = torch.max(f2f, dim=-2)[0] # [B, 256, 256]
        #  att1 = self.atten1(f2f.transpose(0,1)) # TODO: 这里按照MultiHeadAttention的文档，将B和L颠倒
        att1, attention1 = self.atten1(f2f)  # 没有颠倒
        att2, attention2 = self.atten2(att1)
        att3, attention3 = self.atten3(att2)
        att4, attention4 = self.atten4(att3)
        att = torch.cat([att1, att2, att3, att4], dim=-1)
        attention = attention1 + attention2 + attention3 + attention4
        attention = attention / 4 # 这是对x2这些点的点特征，一对多，每个点都对应了其他256个点的特征
        #  print('f2faaaaa', f2f.shape)
        #  print('att,aaa', att.shape)
        att = torch.cat([att, f2f], dim=-1)
        # print('att', att.shape)
        out = self.out(att)
        f_global = torch.max(out, dim=1)[0]  # [B, 1024] 1024是特征长度
        # 在max pooling之前的shape[B, 256, 1024]，在点的维度上max pooling， 参考了pointnet

        return f_global, x2, attention, out, x_feature



def batch_quat2mat(batch_quat):
    '''

    :param batch_quat: shape=(B, 4)
    :return:
    '''
    w, x, y, z = batch_quat[:, 0], batch_quat[:, 1], batch_quat[:, 2], \
                 batch_quat[:, 3]
    device = batch_quat.device
    B = batch_quat.size()[0]
    R = torch.zeros(dtype=torch.float, size=(B, 3, 3)).to(device)
    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return R

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, embed_dim) -> None:
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.embed_dim = embed_dim
        self.head_dix = embed_dim//n_head

        self.w_qs = nn.Linear()

"""
#=============================================
Model
#=============================================
"""
class TouchedRegraster(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Network configs
        self.C = config
        
        # Network componenets
        #  self.Encoder = PCTransformer(config) # TODO: 有个问题，现在的网络是否对旋转敏感？
        #  self.Encoder = PCTransformer_nonsort(config) # TODO: 有个问题，现在的网络是否对旋转敏感？
        self.Encoder = PCTransformer_nonsort(config) # TODO: 有个问题，现在的网络是否对旋转敏感？
        self.Encoder2 = PCTransformer_nonsort(config) # TODO: 有个问题，现在的网络是否对旋转敏感？
        #  self.Encoder2= PCTransformer_nonsort(config)
        # Decoder的结构可能需要多试几次，分别是1，FMR里面的decoder。2，基于transformer的decoder
        #  self.Decoder = PCTransformerDecoder(config) 
        #  self.Decoder = PCTransformerDecoder(config) 
        #  self.mrpcbDecoder = PCTransformerDecoder(config) 
        self.fpc_decoder = BiDecoderNoneCross(config)
        self.rpc_decoder = BiDecoderNoneCross(config)

        # utils for touchedReg
        delta = 1.0e-2  # step size for approx. Jacobian
        dt_initial = torch.autograd.Variable(torch.Tensor([delta for _ in range(6)]))
        self.dt = nn.Parameter(dt_initial.view(1, 6), requires_grad=True)
        self.exp = se3.Exp
        self.transform = se3.transform
        self.inverse = invmat.InvMatrix.apply

        #  self.sort = SortNet()
        #  self.sort = pp.SortNet(1024, 3)
        #  self.sort = SortNetV2(self.C)
        #  self.sort = SortNetV2(self.C)
        
        #  path = os.path.join('/home/code/transReg/models/tst1.ckpt')
        #  model = ma.TouchedRegraster.load_from_checkpoint(path, config=config)
        #  print(model)
        #  model = pl.LightningModule.load_from_checkpoint(path)
        #  self.Encoder = model.Encoder

        self.tfMLP = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 6)
                )

        self.MLPLocalPreRpc = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
                )
        self.MLPLocalPreFpc = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
                )

        self.MLPRpcb = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                )
        self.MLPFpcb = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                )

        if self.C.dataset =='bs':
            fname = open('data/resso/fname.txt', 'r')
            names = fname.readlines()
            self.names = names
            self.normals = np.load('data/resso/bs_norm.npy')
        #  self.fpc_CELoss = nn.CrossEntropyLoss()
        #  self.rpc_CELoss = nn.CrossEntropyLoss()

    def forward(self, batch, bat):
        return self.predict4(batch, bat)

    def predict6(self,batch, batch_indic,need=False, training=False, pretrain=False):
        """
        predict, the first 700 epochs
        """

        if not training:
            self.Encoder.eval()
            self.Encoder2.eval()
            #  self.Encoder2.eval()
            self.tfMLP.eval()
            self.fpc_decoder.eval()
            self.rpc_decoder.eval()
        else:
            self.Encoder.train()
            self.Encoder2.train()
            #  self.Encoder2.train()
            self.tfMLP.train()
            self.fpc_decoder.train()
            self.rpc_decoder.train()
        fpc = batch[0]  # facade point cloud
        mrpc = batch[1] # moved roof pc
        igt = batch[2]  # from rpc to mrpc
        rpc = batch[3]  # roof point cloud
        fpcb=batch[4] # fpc boundary
        mrpcb=batch[5] # mrpc boundary
        batch_size = fpc.shape[0]

        #  print(fpc.shape)

        # maybe for debug purpose
        # 可能要改一下
        #  self.Encoder.train()
        ffpcs = self.Encoder(fpc)
        ffpc = ffpcs[0]
        #  ffpc = self.sort(fpc, ffpc.unsqueeze(1))

        fmrpcs = self.Encoder(mrpc)
        fmrpc = fmrpcs[0]
        #  fmrpc = self.sort(mrpc, fmrpc.unsqueeze(1))

        f = torch.cat([ffpc, fmrpc], dim=-1)

        out = self.tfMLP(f)

        if  pretrain:
            if not need:
                return out
            return out, [0], ffpcs[1], ffpcs[2], fmrpcs[1], fmrpcs[2]

        # decoder for boundary
        de_fpcb = self.Decoder(ffpc)
        de_mrpcb = self.mrpcbDecoder(fmrpc)
        
        if not need:
            return  out, out, de_fpcb, de_mrpcb
        if need:
            return out, [0], ffpcs[1], ffpcs[2], fmrpcs[1], fmrpcs[2], de_fpcb, de_mrpcb # xyz, attention

    

    def predict5(self,batch, batch_indic,need=False, training=False):
        """
        predict
        """

        if not training:
            self.Encoder.eval()
            self.Encoder2.eval()
            #  self.Encoder2.eval()
            self.tfMLP.eval()
            self.fpc_decoder.eval()
            self.rpc_decoder.eval()
        else:
            self.Encoder.train()
            self.Encoder2.train()
            #  self.Encoder2.train()
            self.tfMLP.train()
            self.fpc_decoder.train()
            self.rpc_decoder.train()
        fpc = batch[0]  # facade point cloud
        mrpc = batch[1] # moved roof pc
        igt = batch[2]  # from rpc to mrpc
        rpc = batch[3]  # roof point cloud
        fpcb=batch[4] # fpc boundary
        mrpcb=batch[5] # mrpc boundary
        batch_size = fpc.shape[0]
        fpc_idx = batch[6]
        rpc_idx = batch[7]

        #  print(fpc.shape)

        # maybe for debug purpose
        # 可能要改一下
        #  self.Encoder.train()
        if len(fpc.shape) == 2:
            fpc = fpc.unsqueeze(0)
            mrpc = mrpc.unsqueeze(0)

        ffpcs = self.Encoder(fpc)
        ffpc = ffpcs[0]
        ffpc_local = ffpcs[3]
        #  ffpc = self.sort(fpc, ffpc.unsqueeze(1))
        non_sg_ffpc = ffpcs[4]

        fmrpcs = self.Encoder2(mrpc)
        fmrpc = fmrpcs[0]
        fmrpc_local = fmrpcs[3]
        non_sg_fmrpc = fmrpcs[4]

        #  fmrpc = self.sort(mrpc, fmrpc.unsqueeze(1))

        f = torch.cat([ffpc, fmrpc], dim=-1)

        out = self.tfMLP(f)

        # decoder for boundary
        #  de_fpcb = self.Decoder(ffpc)
        #  de_mrpcb = self.mrpcbDecoder(fmrpc)

        # 原来的做法
        #  de_fpcb = self.fpc_decoder(ffpc_local, ffpc)
        #  de_mrpcb = self.rpc_decoder(fmrpc_local, fmrpc)

        # 交换globalfeature
        #  de_fpcb = self.fpc_decoder(ffpc_local, fmrpc)
        #  de_mrpcb = self.rpc_decoder(fmrpc_local, ffpc)
        non_sg_ffpc = self.MLPLocalPreFpc(non_sg_ffpc) # [B, 1024, 64]
        non_sg_fmrpc = self.MLPLocalPreRpc(non_sg_fmrpc) # [B, 1024, 64]

        non_sg_ffpc_global = torch.max(non_sg_fmrpc, dim=1, keepdim=True)[0]
        non_sg_ffpc_global = non_sg_ffpc_global.repeat(1,1024,1)
        non_sg_fmrpc_global = torch.max(non_sg_fmrpc, dim=1, keepdim=True)[0]
        non_sg_fmrpc_global = non_sg_fmrpc_global.repeat(1,1024,1)

        #  import pdb
        #  pdb.set_trace()
        ffpc_feature4seg = torch.cat([non_sg_fmrpc_global, non_sg_ffpc], dim=-1) # [B, 1024, 128]
        fmrpc_feature4seg = torch.cat([non_sg_ffpc_global, non_sg_fmrpc], dim=-1) # [B, 1024, 128]

        de_fpcb = self.MLPFpcb(ffpc_feature4seg)
        de_fpcb = de_fpcb.permute(0,2,1)
        de_mrpcb = self.MLPRpcb(fmrpc_feature4seg)
        de_mrpcb = de_mrpcb.permute(0,2,1)
        
        if not need:
            return  out, out, de_fpcb, de_mrpcb
        if need:
            return out, [0], ffpcs[1], ffpcs[2], fmrpcs[1], fmrpcs[2], de_fpcb, de_mrpcb # xyz, attention

    def predict4(self,batch, batch_indic,need=False, training=False):
        """
        predict
        """

        if not training:
            self.Encoder.eval()
            self.Decoder.eval()
            self.mrpcbDecoder.eval()
            #  self.Encoder2.eval()
            self.tfMLP.eval()
        else:
            self.Encoder.train()
            self.Decoder.train()
            self.mrpcbDecoder.train()
            #  self.Encoder2.train()
            self.tfMLP.train()
        fpc = batch[0]  # facade point cloud
        mrpc = batch[1] # moved roof pc
        igt = batch[2]  # from rpc to mrpc
        rpc = batch[3]  # roof point cloud
        fpcb=batch[4] # fpc boundary
        mrpcb=batch[5] # mrpc boundary
        batch_size = fpc.shape[0]

        #  print(fpc.shape)

        # maybe for debug purpose
        # 可能要改一下
        #  self.Encoder.train()
        ffpcs = self.Encoder(fpc)
        ffpc = ffpcs[0]
        #  ffpc = self.sort(fpc, ffpc.unsqueeze(1))

        fmrpcs = self.Encoder(mrpc)
        fmrpc = fmrpcs[0]
        #  fmrpc = self.sort(mrpc, fmrpc.unsqueeze(1))

        f = torch.cat([ffpc, fmrpc], dim=-1)

        out = self.tfMLP(f)

        # decoder for boundary
        de_fpcb = self.Decoder(ffpc, ffpcs[1])
        de_mrpcb = self.mrpcbDecoder(fmrpc, fmrpcs[1])
        
        if not need:
            return  out, out, de_fpcb, de_mrpcb
        if need:
            return out, [0], ffpcs[1], ffpcs[2], fmrpcs[1], fmrpcs[2], de_fpcb, de_mrpcb # xyz, attention

    def predict3(self,batch, batch_indic,need=False, training=False):
        """
        predict
        """

        if not training:
            self.Encoder.eval()
            #  self.Encoder2.eval()
            self.tfMLP.eval()
        else:
            self.Encoder.train()
            #  self.Encoder2.train()
            self.tfMLP.train()
        fpc = batch[0]  # facade point cloud
        mrpc = batch[1] # moved roof pc
        igt = batch[2]  # from rpc to mrpc
        rpc = batch[3]  # roof point cloud
        batch_size = fpc.shape[0]

        #  print(fpc.shape)

        # maybe for debug purpose
        # 可能要改一下
        #  self.Encoder.train()
        ffpcs = self.Encoder(fpc)
        ffpc = ffpcs[0]
        #  ffpc = self.sort(fpc, ffpc.unsqueeze(1))

        fmrpcs = self.Encoder(mrpc)
        fmrpc = fmrpcs[0]
        #  fmrpc = self.sort(mrpc, fmrpc.unsqueeze(1))

        f = torch.cat([ffpc, fmrpc], dim=-1)

        out = self.tfMLP(f)
        
        if not need:
            return  out, out
        if need:
            return out, [0], ffpcs[1], ffpcs[2], fmrpcs[1], fmrpcs[2] # xyz, attention


    def predict2(self,batch, batch_indic,need=False, training=False):
        """
        predict
        """

        if not training:
            self.Encoder.eval()
            #  self.Encoder2.eval()
            self.tfMLP.eval()
        else:
            self.Encoder.train()
            #  self.Encoder2.train()
            self.tfMLP.train()
        fpc = batch[0]  # facade point cloud
        mrpc = batch[1] # moved roof pc
        igt = batch[2]  # from rpc to mrpc
        rpc = batch[3]  # roof point cloud
        batch_size = fpc.shape[0]

        # maybe for debug purpose
        # 可能要改一下
        #  self.Encoder.train()
        ffpcs = self.Encoder(fpc)
        ffpc = ffpcs[0]
        #  ffpc = self.sort(fpc, ffpc.unsqueeze(1))

        fmrpcs = self.Encoder(mrpc)
        fmrpc = fmrpcs[0]
        #  fmrpc = self.sort(mrpc, fmrpc.unsqueeze(1))

        f = torch.cat([ffpc, fmrpc], dim=-1)

        out = self.tfMLP(f)
        
        t = out[:,:3]
        q = out[:, 3:] / torch.norm(out[:, 3:], dim=1, keepdim=True)

        R = batch_quat2mat(q)
        if not need:
            return  R, t
        if need:
            return R, t, ffpcs[1], ffpcs[2], fmrpcs[1], fmrpcs[2] # xyz, attention

    def vis(self, tag, pc1, pc2, index, logger):
        pc = torch.cat([pc1[index], pc2[index]], axis=0)
        color1 = torch.from_numpy(np.array([1, 0.706, 0])*255).expand(pc1.shape[1], 3)
        color2 = torch.from_numpy(np.array([0, 0.651, 0.929])*255).expand(pc2.shape[1], 3)
        color = torch.cat([color1, color2], axis=0)
        logger.experiment.add_mesh(tag, pc.unsqueeze(0), color.unsqueeze(0), global_step=self.global_step)

    def on_train_start(self) -> None:
        self.logger.experiment.add_text('message', self.C.m)
        self.logger.experiment.add_text('path', self.C.output_path)
        self.logger.experiment.add_text('loss_mode', str( self.C.loss_mode) )
        self.logger.experiment.add_text('file', str(__file__) )
        self.logger.experiment.add_text('lr', str(self.C.lr) )
        return super().on_train_start()

    def training_step(self, batch, batch_indic):
        """
        forward training
        """
        #  self.Encoder.eval()
        fpc = batch[0]  # facade point cloud
        mrpc = batch[1] # moved roof pc
        igt = batch[2]  # from rpc to mrpc
        rpc = batch[3]  # roof point cloud
        fpcb = batch[4]
        rpcb = batch[5]
        batch_size = fpc.shape[0]
        fpc_idx = batch[6]
        rpc_idx = batch[7]

        
        pretrain = self.current_epoch < self.C.pretrain_epochs
        if pretrain:
            # 只训练mlp一个branch
            out, _, x2, attention, mrpc_x2, mrpc_attention = self.predict6(batch, batch_size, training=True, need=True, pretrain=True)
        elif not pretrain:
            out, t, x2, attention, mrpc_x2, mrpc_attention, de_fpcb, de_mrpcb = self.predict5(batch, batch_size, training=True, need=True)



        att1 = attention.mean(dim=1)
        att2 = mrpc_attention.mean(dim=1)
        top64att1 = torch.topk(att1, 32)
        x2att1=x2[:, top64att1[1][:,0]]
        top64att2 = torch.topk(att2, 32)
        x2att2=mrpc_x2[:, top64att2[1][:, 0]]



        #  de_mrpc = self.trans(mrpc, R, t)
        mat = se3.exp(out).to(mrpc)
        de_mrpc = se3.transform(se3.exp(out).to(mrpc), mrpc.permute(0,2,1))
        de_mrpc = de_mrpc.permute(0,2,1)
        #  print(de_mrpc.shape)
        R = mat[:, :3, :3]
        t = mat[:, :3, 3]
        #  print(de_mrpc.shape)
        #  print(rpc.shape)

        dg_mrpc_dist1, dg_mrpc_dist2 = self.chamfer_loss(rpc, de_mrpc)
        if self.C.loss_sum:
            loss_recoversy = torch.sum(dg_mrpc_dist1) + torch.sum(dg_mrpc_dist2)
        else:
            loss_recoversy = torch.mean(dg_mrpc_dist1) + torch.mean(dg_mrpc_dist2)


        g = torch.eye(4).unsqueeze(0).repeat(R.shape[0], 1, 1).to(R)
        g[:, :3, :3] = R
        g[:, :3, 3] = t

        loss_g = self.comp(g, igt)

        #  self.log('train/loss_r', loss_r)
        #  self.log('train/loss_g', loss_g)
        #  self.log('train/loss_recoversy', loss_recoversy)
        self.log('train/loss_re', loss_recoversy)
        self.log('train/loss_g', loss_g)
        index = 1
        self.vis('train_output', fpc, de_mrpc, index, self.logger)
        self.vis('train_dataset', fpc, mrpc, index, self.logger)
        self.vis('train_gt', fpc, rpc, index, self.logger)
        self.log('lr', self.scheduler.get_last_lr()[0])
        #  print(x2.shape)
        self.vis_attention('train_fpc_attention_map', x2, attention, index)
        self.vis_attention('train_mrpc_attention_map', mrpc_x2, mrpc_attention, index)
        self.vis('train__x2', x2, mrpc_x2, index, self.logger)
        

        #  # unsupervised learning
        #  if self.C.loss_mode == 0:
            #  loss = loss_task1
        #  elif self.C.loss_mode == 1:
            #  #  loss = loss_task1 + loss_g
            #  loss = self.C.alpha*loss_task1 + self.C.beta*loss_g
        #  elif self.C.loss_mode == 2:
            #  loss = loss_r
        #  elif self.C.loss_mode == 3:
            #  loss = loss_g
        #  elif self.C.loss_mode == 4:
            #  loss = loss_task1 + loss_recoversy
        #  elif self.C.loss_mode == 5:
            #  loss = loss_recoversy
        

        dg_att1_dist1, dg_att1_dist2 = self.chamfer_loss(x2att1, x2att2)
        emd = earth_mover_distance( de_mrpc, rpc, transpose=False)
        if self.C.loss_sum:
            loss_emd = torch.sum(emd)
            loss_cd2 = torch.sum(dg_att1_dist1) + torch.sum(dg_att1_dist2)
            self.log('train/cd2', loss_cd2)
        else:
            loss_emd = torch.mean(emd)
            loss_cd2 = torch.mean(dg_att1_dist1) + torch.mean(dg_att1_dist2)
            self.log('train/cd2', loss_cd2)

        emd2 = earth_mover_distance(x2att1, x2att2, transpose=False)
        

        self.log('train/loss_emd', loss_emd)
        if self.C.loss_mode == 0:
            loss = loss_recoversy + loss_g
        elif self.C.loss_mode == 1:
            loss = loss_recoversy + loss_g + loss_emd
        elif self.C.loss_mode == 2:
            loss = loss_emd
        elif self.C.loss_mode == 3:
            loss = loss_emd + loss_g
        elif self.C.loss_mode == 4:
            loss = loss_emd + loss_recoversy
        elif self.C.loss_mode == 5:
            loss = loss_g
        elif self.C.loss_mode == 6:
            loss = loss_recoversy
        #  elif self.C.loss_mode == 6:
            #  loss = 0

        if self.C.loss_sum:
            emd2 = torch.sum(emd2)
        elif not self.C.loss_sum:
            emd2 = torch.sum(emd2)
        self.log('train_emd2', emd2)

        if self.C.use_emd2:
            loss += emd2

        if self.C.use_cd2:
            loss += loss_cd2


        if pretrain:
            self.log('train_loss', loss)
            return {'loss':loss}

        de_fpcb_idx = de_fpcb
        de_mrpcb_idx = de_mrpcb # [B, 2, 1024]

        # CrossEntroyLoss

        #  print(de_fpcb_idx.shape)
        #  print(fpc_idx.shape)
        #  flatten_de_fpcb_idx = de_fpcb_idx.flatten(0,1)
        #  flatten_fpc_idx = fpc_idx.flatten(0,1)
        #  loss_fpcb_cel = F.cross_entropy(de_fpcb_idx.flatten(0,1), fpc_idx.squeeze(-1).flatten(0,1).to(torch.int64))
        #  loss_rpcb_cel = F.cross_entropy(de_mrpcb_idx.flatten(0,1), rpc_idx.squeeze(-1).flatten(0,1).to(torch.int64))

        #  pdb.set_trace()
        loss_fpcb_cel = F.cross_entropy(de_fpcb_idx, fpc_idx.squeeze().long())
        loss_rpcb_cel = F.cross_entropy(de_mrpcb_idx, rpc_idx.squeeze().long())

        #  loss_fpcb_cel = F.log_softmax(de_fpcb_idx.transpose(2,1).view(-1, 2), dim=-1)
        #  #  print(loss_fpcb_cel.shape)
        #  #  print(fpc_idx.view(-1,1).shape)
        #  loss_fpcb_cel = F.nll_loss(loss_fpcb_cel, fpc_idx.view(-1, 1).long()[:, 0])
        #  loss_rpcb_cel = F.log_softmax(de_mrpcb_idx.transpose(2,1).view(-1, 2), dim=-1)
        #  loss_rpcb_cel = F.nll_loss(loss_rpcb_cel, rpc_idx.view(-1, 1).long()[:, 0])

        #  print(loss_fpcb_cel.shape)
        #  print(loss_rpcb_cel.shape)

        loss += loss_fpcb_cel
        loss += loss_rpcb_cel

        self.log('train/loss_fpcb_cel', loss_fpcb_cel)
        self.log('train/loss_rpcb_cel', loss_rpcb_cel)

        #  de_fpcb_idx_sig = torch.sigmoid(de_fpcb_idx)
        #  de_mrpcb_idx_sig = torch.sigmoid(de_mrpcb_idx)

        de_fpcb_idx_sig = torch.softmax(de_fpcb_idx, dim=1)  # [B, 2, 1024]
        de_mrpcb_idx_sig = torch.softmax(de_mrpcb_idx, dim=1) # [B, 2, 1024]

        de_fpcb_idx_sig = de_fpcb_idx_sig[:, 1, :] # [B, 1024]
        de_fpcb_idx = torch.topk(de_fpcb_idx_sig, 128, 1)[1] # [B, 128]
        de_mrpcb_idx_sig = de_mrpcb_idx_sig[:, 1, :]
        de_mrpcb_idx = torch.topk(de_mrpcb_idx_sig, 128, 1)[1]

        # Calculate the iou
        pred_1_fpc = torch.zeros(fpc_idx.shape[0], 1024).to(fpc_idx)
        pred_1_fpc = pred_1_fpc.scatter(1, de_fpcb_idx, 1)
        pred_1_mrpc = torch.zeros(fpc_idx.shape[0], 1024).to(fpc_idx)
        pred_1_mrpc = pred_1_mrpc.scatter(1, de_mrpcb_idx, 1) #[B, 1024]
        fpcb_I = torch.sum(torch.logical_and(pred_1_fpc, fpc_idx)).float()
        fpcb_U = torch.sum(torch.logical_or(pred_1_fpc, fpc_idx)).float()
        fpc_iou = fpcb_I / fpcb_U
        mrpcb_I = torch.sum(torch.logical_and(pred_1_mrpc, rpc_idx)).float()
        mrpcb_U = torch.sum(torch.logical_or(pred_1_mrpc, rpc_idx)).float()
        mrpcb_iou = mrpcb_I / mrpcb_U
        self.log('train/fpc_iou', fpc_iou)
        self.log('train/mrpcb_iou', mrpcb_iou)

        #  de_fpcb = fpc[de_fpcb_idx]
        #  de_mrpcb = mrpc[de_mrpcb_idx]
        de_fpcb = torch.gather(fpc, 1, de_fpcb_idx.unsqueeze(-1).repeat(1,1,3))
        de_mrpcb = torch.gather(mrpc, 1, de_mrpcb_idx.unsqueeze(-1).repeat(1,1,3))

        cd_fpcb1, cd_fpcb2 = self.chamfer_loss(de_fpcb, fpcb)
        loss_fpcb =  torch.mean(cd_fpcb1) + torch.mean(cd_fpcb2)
        self.log('train/loss_fpcb', loss_fpcb)

        inverse_de_mrpcb = se3.transform(se3.exp(out).to(mrpc), de_mrpcb.permute(0,2,1))
        inverse_de_mrpcb = inverse_de_mrpcb.permute(0,2,1)

        cd_mrpcb1, cd_mrpcb2 = self.chamfer_loss(inverse_de_mrpcb,  rpcb)
        loss_mrpcb = torch.mean(cd_mrpcb1) + torch.mean(cd_mrpcb2)
        self.log('train/loss_rpcb', loss_mrpcb)

        emd_fpcb = earth_mover_distance(de_fpcb, fpcb, transpose=False)
        emd_fpcb = torch.mean(emd_fpcb)
        emd_mrpcb = earth_mover_distance(inverse_de_mrpcb, rpcb, transpose=False)
        emd_mrpcb = torch.mean(emd_mrpcb)
        self.log('train/loss_emd_fpcb', emd_fpcb)
        self.log('train/loss_emc_mrpcb', emd_mrpcb)

        self.vis('train_mrpcb',  rpcb, inverse_de_mrpcb, index, self.logger)
        #  self.vis('train_pcb_dataset', mrpc, mrpcb, index, self.logger)
        #  self.vis('train_fpcb_mrpcb', fpcb, de_mrpcb, index, self.logger)
        self.vis('train_mrpcb_fpcb', de_fpcb,inverse_de_mrpcb, index, self.logger)
        self.vis('train_mrpcb_fpcb', fpcb, rpcb, index, self.logger)

        #  self.vis('train_fpcb_invdemrpcb', fpcb, inverse_de_mrpcb, index, self.logger)
        #  cd_inverse_demrpcb1, cd_inverse_demrpcb2 = self.chamfer_loss(inverse_de_mrpcb, mrpcb)
        #  loss_inverse_demrpcb = torch.mean(cd_inverse_demrpcb1) + torch.mean(cd_inverse_demrpcb2)
        #  self.log('train/loss_inverse_demrpcb', loss_inverse_demrpcb)

        #  loss += loss_inverse_demrpcb

        #  emd_inverse_mrpcb = earth_mover_distance(inverse_de_mrpcb, )


        loss += loss_mrpcb
        loss += loss_fpcb

        if self.C.use_emd3:
            loss += emd_fpcb
            loss += emd_mrpcb


        self.log('train_loss', loss)
        return  {'loss': loss}

    def training_epoch_end(self, outputs):
        epoch_losses = []
        for i in outputs:
            epoch_losses.append(i['loss'].clone().detach().unsqueeze(0))
        epoch_loss =  torch.cat(epoch_losses, dim=0)
        epoch_loss = torch.mean(epoch_loss)
        self.log('epoch_loss', epoch_loss)
        #  print('hello')
        if epoch_loss < 7:
            #  print('xiaoyu')
            with open(os.path.join(self.C.output_path, 'stop.txt'), 'w+') as f:
                f.writelines('stop\n')
                f.writelines(str(self.current_epoch)+'\n')
                f.writelines(str(epoch_loss))

    def validation_step(self, batch, batch_size):
        # predict
        #  g = torch.randn(batch[0].shape[0], 6,6)
        #  g = self.predict(batch, batch_size)

        pretrain = self.current_epoch < self.C.pretrain_epochs
        if pretrain:
            # 只训练mlp一个branch
            out = self.predict6(batch, batch_size, training=False, need=False, pretrain=True)
        elif not pretrain:
            out, t,  de_fpcb, de_mrpcb = self.predict5(batch, batch_size, training=False, need=False)

        # prepare data
        fpc = batch[0]  # [B, N, 3]
        mrpc = batch[1]
        igt = batch[2]  # [B, ] from p0 to p1
        rpc = batch[3]
        fpcb = batch[4]
        rpcb = batch[5]
        fpc_idx = batch[6]
        rpc_idx = batch[7]

        #  print(fpc.shape)
        #  g = g.cpu()
        #  fpc = fpc.cpu()
        #  print(g.shape)
        #  print(fpc.shape)

        # transform
        #  valfpc = self.transform(g.cpu().unsqueeze(1), fpc.cpu())
        #  valrpc = self.trans(mrpc, R, t)
        valrpc = se3.transform(se3.exp(out).to(mrpc), mrpc.permute(0,2,1))
        #  print(valrpc.shape)
        valrpc = valrpc.permute(0,2,1)
        mat = se3.exp(out).to(mrpc)
        R = mat[:, :3, :3]
        t = mat[:, :3, 3]




        # task: draw on tensorboard
        index  = 1
        color1 = torch.from_numpy(np.array([1, 0.706, 0])*255).expand(1024, 3)
        color2 = torch.from_numpy(np.array([0, 0.651, 0.929])*255).expand(1024, 3)
        color = torch.cat([color1, color2], axis=0)
        points = torch.cat([valrpc[index].cpu(), fpc.cpu()[index]], axis = 0)
        self.logger.experiment.add_mesh('实验结果', points.unsqueeze(0), color.unsqueeze(0), global_step=self.global_step)

        # 还没移
        pointsgt = torch.cat([rpc[index], fpc[index]], axis=0)
        self.logger.experiment.add_mesh('gt', pointsgt.unsqueeze(0), color.unsqueeze(0), global_step=self.global_step)

        # 数据集里的样子
        points_f_mr = torch.cat([mrpc[index], fpc[index]], axis=0)
        self.logger.experiment.add_mesh('数据集', points_f_mr.unsqueeze(0), color.unsqueeze(0), global_step=self.global_step)

        s = self.compute_metrics(R, t, igt)
        scores = []
        for i in s:
            scores.append(torch.mean(torch.tensor(i)))
        self.log('val/r_mse', scores[0])
        self.log('val/r_mae', scores[1])
        self.log('val/t_mse', scores[2])
        self.log('val/t_mae', scores[3])
        self.log('val/r_isotropic', scores[4])
        self.log('val/t_isotropic', scores[5])


        if not pretrain:
            de_fpcb_idx = de_fpcb
            de_mrpcb_idx = de_mrpcb

            de_fpcb_idx_sig = torch.softmax(de_fpcb_idx, dim=1)
            de_mrpcb_idx_sig = torch.softmax(de_mrpcb_idx, dim=1)


            de_fpcb_idx_sig = de_fpcb_idx_sig[:, 1, :] # [B, 1024]
            de_fpcb_idx = torch.topk(de_fpcb_idx_sig, 128, 1)[1] # [B, 128]
            de_mrpcb_idx_sig = de_mrpcb_idx_sig[:, 1, :]
            de_mrpcb_idx = torch.topk(de_mrpcb_idx_sig, 128, 1)[1]

            # Calculate the iou
            pred_1_fpc = torch.zeros(fpc_idx.shape[0], 1024).to(de_fpcb_idx)
            pred_1_fpc = pred_1_fpc.scatter(1, de_fpcb_idx, 1)
            pred_1_mrpc = torch.zeros(fpc_idx.shape[0], 1024).to(de_mrpcb_idx)
            pred_1_mrpc = pred_1_mrpc.scatter(1, de_mrpcb_idx, 1) #[B, 1024]
            fpcb_I = torch.sum(torch.logical_and(pred_1_fpc, fpc_idx)).float()
            fpcb_U = torch.sum(torch.logical_or(pred_1_fpc, fpc_idx)).float()
            fpc_iou = fpcb_I / fpcb_U
            mrpcb_I = torch.sum(torch.logical_and(pred_1_mrpc, rpc_idx)).float()
            mrpcb_U = torch.sum(torch.logical_or(pred_1_mrpc, rpc_idx)).float()
            mrpcb_iou = mrpcb_I / mrpcb_U
            self.log('val/fpc_iou', fpc_iou)
            self.log('val/mrpcb_iou', mrpcb_iou)

            #  de_fpcb = fpc[de_fpcb_idx]
            #  de_mrpcb = mrpc[de_mrpcb_idx]
            de_fpcb = torch.gather(fpc, 1, de_fpcb_idx.unsqueeze(-1).repeat(1,1,3))
            de_mrpcb = torch.gather(mrpc, 1, de_mrpcb_idx.unsqueeze(-1).repeat(1,1,3))
            # 看看边界移的怎么样
            de_mrpcb = se3.transform(se3.exp(out).to(mrpc), de_mrpcb.permute(0,2,1))
            de_mrpcb = de_mrpcb.permute(0, 2, 1)
            self.vis('val_rpcb&fpcb', rpcb, fpcb, index, self.logger)
            self.vis('val_de_mrpcb&fpcb', de_mrpcb, fpcb, index, self.logger)
            self.vis('val_de_mrpcb&de_fpcb', de_mrpcb, de_fpcb, index, self.logger)

    def test_step(self, batch, batch_id):

        new_batch = []
        for b in batch[:-2]:
            if len(b.shape) == 2:
                newb=b.unsqueeze(0)
                new_batch.append(newb)
            else:
                new_batch.append(b)
        for b in batch[-2:]:
            if len(b.shape) == 1:
                newb = b.unsqueeze(0)
                new_batch.append(newb)
            else:
                new_batch.append(b)
        
        fpc = new_batch[0]  # [B, N, 3]
        mrpc = new_batch[1]
        igt = new_batch[2]  # [B, 4,4,] from p0 to p1
        rpc = new_batch[3]
        fpcb = new_batch[4] # [B, 1024, 3] boundary of the fpc and rpc
        rpcb = new_batch[5]
        fpc_idx = new_batch[6] # [B, 1024]
        rpc_idx = new_batch[7]

        # test shape in test batch
        #  for i in new_batch:
            #  print(i.shape)

        batch_size = fpc.shape[0]

        # 测配准部分的结果
        out, _, de_fpcb, de_mrpcb = self.predict5(new_batch, batch_size, training=False, need=False) 
        #  import pdb
        #  pdb.set_trace()
        mat = se3.exp(out).to(self.device)
        R = mat[:, :3, :3]
        t = mat[:, :3, 3]
        s = self.compute_metrics(R, t, igt)
        scores = []
        for i in s:
            scores.append(torch.mean(torch.tensor(i)))

        # 测边界的结果
        de_fpcb_idx = de_fpcb
        de_mrpcb_idx = de_mrpcb # [B, 2, 1024]

        de_fpcb_idx_sig = torch.softmax(de_fpcb_idx, dim=1)  # [B, 2, 1024]
        de_mrpcb_idx_sig = torch.softmax(de_mrpcb_idx, dim=1) # [B, 2, 1024]

        de_fpcb_idx_sig = de_fpcb_idx_sig[:, 1, :] # [B, 1024]
        de_fpcb_idx = torch.topk(de_fpcb_idx_sig, 128, 1)[1] # [B, 128]
        de_mrpcb_idx_sig = de_mrpcb_idx_sig[:, 1, :]
        de_mrpcb_idx = torch.topk(de_mrpcb_idx_sig, 128, 1)[1]

        # Calculate the iou
        pred_1_fpc = torch.zeros(fpc_idx.shape[0], 1024).to(fpc_idx)
        pred_1_fpc = pred_1_fpc.scatter(1, de_fpcb_idx, 1)
        pred_1_mrpc = torch.zeros(fpc_idx.shape[0], 1024).to(fpc_idx)
        pred_1_mrpc = pred_1_mrpc.scatter(1, de_mrpcb_idx, 1) #[B, 1024]
        fpcb_I = torch.sum(torch.logical_and(pred_1_fpc, fpc_idx)).float()
        fpcb_U = torch.sum(torch.logical_or(pred_1_fpc, fpc_idx)).float()
        fpc_iou = fpcb_I / fpcb_U
        mrpcb_I = torch.sum(torch.logical_and(pred_1_mrpc, rpc_idx)).float()
        mrpcb_U = torch.sum(torch.logical_or(pred_1_mrpc, rpc_idx)).float()
        mrpcb_iou = mrpcb_I / mrpcb_U

        scores.append(fpc_iou)
        scores.append(mrpcb_iou)

        # calculate boundary cd
        #  import pdb; pdb.set_trace()
        de_fpcb = torch.gather(fpc, 1, de_fpcb_idx.unsqueeze(-1).repeat(1,1,3))
        cd_fpc1, cd_fpc2 = self.chamfer_loss(fpcb, de_fpcb)
        cd_fpc = torch.mean(cd_fpc1) + torch.mean(cd_fpc2)
        de_rpcb = torch.gather(rpc, 1, de_mrpcb_idx.unsqueeze(-1).repeat(1,1,3))
        de_rpcb = se3.transform(mat, de_rpcb.permute(0,2,1)).permute(0,2,1)
        #  import pdb; pdb.set_trace()
        cd_rpc1, cd_rpc2 = self.chamfer_loss(rpcb, de_rpcb)
        cd_rpc = torch.mean(cd_rpc1) + torch.mean(cd_rpc2)
        scores.append(cd_fpc)
        scores.append(cd_rpc)

        return torch.tensor(scores, device=self.device).unsqueeze(0)

    def test_epoch_end(self, outputs) -> None:
        #  print('yes')
        #  print(len(outputs))
        #  print(len(outputs[0]))
        #  print(outputs[0])
        #  print(type(outputs[0]))
        #  outputs = torch.tensor(outputs, device=self.device)
        outputs = torch.cat(outputs, dim=0)
        s = torch.mean(outputs, dim=0)
        name = ['r_mse','   r_mae','   t_mse','    t_mae','    r_iso','    t_iso','  fpc_iou','   mrpc_iou',' cd_fpcb',' cd_rpcb']
        print(s)
        for i in range(10):
            print(name[i]+'   ')
            print(s[i])
        with open(os.path.join(self.C.output_path, datetime.datetime.now().strftime('%b%d_%H-%M-%S')+'metrics.txt'), 'w+') as fout:
            fout.writelines('r_mse,   r_mae,   t_mse,    t_mae,    r_iso,    t_iso,  fpc_iou,   mrpc_iou, cd_fpcb, cd_rpcb \n')
            for ss in s:
                fout.writelines(str(ss.detach().data.cpu().numpy())+'   ')
            fout.writelines('\n')


    def vis_attention(self, tag, x2, attention, index,  axis=True):
        """
        attention:  [B, 255, 255]
        """
        coordi = x2.detach().cpu()[index] # [B, n, 3] n=256, 看第二次的sg
        fig = plt.figure(dpi=100, frameon=False)
        ax = fig.gca(projection ='3d')
        if not axis:
            plt.axis('off')
        attention = attention.detach().cpu()
        attention = attention[index]
        attention = attention.mean(dim=0)
        the_fourth_dimension = attention.numpy()
        the_fourth_dimension = (the_fourth_dimension-min(the_fourth_dimension))/(max(the_fourth_dimension)-min(the_fourth_dimension))
        colors = cm.cividis(the_fourth_dimension)

        ax.scatter(coordi[:, 0], coordi[:, 1], coordi[:, 2], c=colors, marker='o', s=10)

        colmap = cm.ScalarMappable(cmap=cm.cividis)
        colmap.set_array(the_fourth_dimension)
        fig.colorbar(colmap)
        self.logger.experiment.add_figure(tag, fig, global_step=self.global_step)
        return fig



    def split(self, filename, normal):
        pc = o3d.io.read_point_cloud(filename.replace('''\\''', '''/'''))
        points = np.asarray(pc.points)
        dis = np.dot(points, normal)
        bool = np.array([dis>=0]).squeeze(0).squeeze(1)
        up = points[bool]
        bool = np.array([dis<0]).squeeze(0).squeeze(1)
        down = points[bool]

        return up, down





    def compute_metrics(self, R, t, igt):
        """
        R,      [B, 3, 3]   predicted R
        t,      [B, 3]      predicted t
        igt,    [B, 4, 4]   gt
        """
        gtR = igt[:, :3, :3]
        gtt = igt[:, :3, 3]
        inv_R, inv_t = metrics.inv_R_t(gtR, gtt)
        cur_r_mse, cur_r_mae = metrics.anisotropic_R_error(R, inv_R)
        cur_t_mse, cur_t_mae = metrics.anisotropic_t_error(t, inv_t)
        cur_r_isotropic = metrics.isotropic_R_error(R, inv_R)
        #  cur_r_isotropic = metrics.isotropic_R_error(R, gtR)
        cur_t_isotropic = metrics.isotropic_t_error(t, inv_t, inv_R)
        return cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, cur_t_isotropic





    def trans(self, pc, R, t):
        #  print(R.shape)
        #  print(pc.shape)
        #  print(t.shape)
        return pc.bmm(R) + t.unsqueeze(1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.C.lr)
        #  optimizer = torch.optim.SGD(self.parameters(), lr=self.C.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.999)
        return [ optimizer ], [{'scheduler':self.scheduler, 'interval':'step'}]

    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    # estimate Jacobian matrix
    def approx_Jac(self, p0, f0, dt):
        # p0: [B, N, 3], Variable
        # f0: [B, K], correspanding feature Vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]
        batch_size = p0.size(0)
        num_points = p0.size(1)

        # compute transforms
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :]) # [6, 6]
            D = self.exp(-d) # [6, 4, 4]
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous() # [B, 6, 1, 4, 4]
        p = self.transform(transf, p0.unsqueeze(1))
        
        f0 = f0.unsqueeze(-1)
        f1 = self.Encoder(p.view(-1, num_points, 3))[0]
        f1 = self.sort(p.view(-1, num_points, 3),f1.unsqueeze(1)) # 新家的
        #  print('f1', f1.shape)
        f = f1.view(batch_size, 6, -1).transpose(1,2)

        df = f0 - f # [B, K, 6]
        J = df / dt.unsqueeze(1)

        return J



    def chamfer_loss(self, a, b):
        x, y = a, b
        bs, numpoints, pc_dim = x.size()
        xx = torch.bmm(x, x.transpose(2,1))
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        diag_ind = torch.arange(0, numpoints)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2,1) + ry - 2*zz)
        return torch.min(P, 1)[0], torch.min(P, 2)[0]

    def rsq(self, r):
        # |r| should be 0
        z = torch.zeros_like(r)
        return F.mse_loss(r, z, reduction='sum')

    def comp(self, g, igt):
        """|g*igt - I | (should be 0)"""
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1,4,4).repeat(A.size(0), 1, 1)
        return F.mse_loss(A, I, reduction='mean')*16


def parse_arg(argv=None):
    parser = argparse.ArgumentParser('touched regis')

    # required args
    parser.add_argument('--output_path', help='path to store output_folders', type=str, 
            default='TRG')
    parser.add_argument('--device', help='choose device to use', type=int, default=2)
    parser.add_argument('--dim-k', default=1024, type=int, help='length of feature')
    parser.add_argument('--epochs', default=100000, type=int, help='epochs')
    parser.add_argument('--max_iter', default=50, type=int, help='max-iter on IC algorithm')
    parser.add_argument('--dt', help='data type of input, default=fr', type=str, default='fr')
    parser.add_argument('--mode', help='mode for now', default='train')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--save_on_epochs', default=1, type=int, help='save on how many epochs')
    parser.add_argument('--loss_mode', default=0, type=int, help='mode of loss using')
    parser.add_argument('--valp', default=0, type=int, help='how often check valuation per epochs')
    parser.add_argument('-m', help='leave a message for this round of training/valuation', type=str, 
            default='default string')
    parser.add_argument('--loss_sum', default=False, action='store_true', help="是使用sum还是mean计算loss，sum default")
    parser.add_argument('--pretrain_epochs', default=700, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--dataset', default='fr', type=str, help='dataset type, fr or cad or bs')
    parser.add_argument('--look', default=False, action='store_true', help='look or not')
    parser.add_argument('--random', default=False, action='store_true', help='use random dataset')
    parser.add_argument('--use_emd2', action='store_true', help='look or not', default=False)
    parser.add_argument('--use_cd2', action='store_true', help='look or not', default=False)
    parser.add_argument('--use_emd3', action='store_true', help='use emd as loss for fpcb', default=False)
    parser.add_argument('--random_slice', action='store_true', help='look or not', default=False)
    parser.add_argument('--verbose', action='store_true', help='verbose or not', default=False)


    args = parser.parse_args(argv)
    return args

def per_parse(opt):
    if opt.output_path == 'TRG':
        cpath = os.path.join(os.path.dirname(__file__), 'TRG')
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        dpath = os.path.join(cpath, datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
        opt.output_path = dpath
    opt.swap_axis = True



def main(opt):
    # save setting
    argsDic = opt.__dict__
    dir = os.path.join(opt.output_path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(os.path.join(dir, 'setting.txt'), 'w+') as f:
        f.writelines('-----------starting---------------\n')
        f.writelines(__file__+ '\n')
        f.writelines(time.asctime(time.localtime(time.time()))+'\n')
        for eachArg, value in argsDic.items():
            f.writelines(eachArg+"   "*8+str(value)+'\n'  )
        f.writelines('---------------ending-------------\n')

    print('************************************')
    print('     ', opt.output_path)
    print('************************************')


    # get Model
    model = TouchedRegraster(opt)

    # get train and val datasets
    if opt.dataset == 'fr':
        traindataset, tstdataset = dataset.get_datasets()
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True, 
                num_workers=torch.cuda.device_count()*4)
    elif opt.dataset == 'cad':
        # ModelNet数据集
        traindataset, tstdataset = dataset.get_cad_datasets()
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True, 
                num_workers=torch.cuda.device_count()*4)
    elif opt.dataset == 'cadrr2':
        # ModelNet数据集
        traindataset, tstdataset = dataset.get_cad_datasets(opt.dataset, random=opt.random, random_slice=opt.random_slice)
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True, 
                num_workers=torch.cuda.device_count()*4)
    elif opt.dataset == 'cadrr':
        # ModelNet数据集
        traindataset, tstdataset = dataset.get_cad_datasets(opt.dataset, random=opt.random)
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True, 
                num_workers=torch.cuda.device_count()*4)
    elif opt.dataset == 'cadr':
        # ModelNet数据集
        traindataset, tstdataset = dataset.get_cad_datasets(opt.dataset, random=opt.random)
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True, 
                num_workers=torch.cuda.device_count()*4)
    elif opt.dataset == 'bs':
        print('yes')
        traindataset, tstdataset = dataset.get_cad_datasets(opt.dataset)
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True, 
                num_workers=torch.cuda.device_count()*4)
    elif opt.dataset == 'snp':
        print('yes')
        traindataset, tstdataset = dataset.get_cad_datasets(opt.dataset)
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True, 
                num_workers=torch.cuda.device_count()*4)
    elif opt.dataset == 'cadpro':
        # ModelNet数据集
        traindataset, tstdataset = dataset.get_cad_datasets(category='cadpro')
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True, 
                num_workers=torch.cuda.device_count()*4)
    else:
        traindataset, _, tstdataset = dataset.get_dataset(opt.dataset)
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=True,
                num_workers=torch.cuda.device_count()*4)
        tst_loader = torch.utils.data.DataLoader(tstdataset, batch_size=opt.batch_size, drop_last=True,
                num_workers=torch.cuda.device_count()*4)


    # get test datasets
    if opt.dataset == 'cadrr':
        true_testdataset = dataset.get_test_dataset(opt.dataset, random=opt.random)
        true_testdataloader = torch.utils.data.DataLoader(true_testdataset, batch_size=opt.batch_size, drop_last=False, shuffle=False, num_workers=torch.cuda.device_count()*4)
    elif opt.dataset == 'cadrr2':
        true_testdataset = dataset.get_test_dataset(opt.dataset, random=opt.random, random_slice=opt.random_slice)
        true_testdataloader = torch.utils.data.DataLoader(true_testdataset, batch_size=opt.batch_size, drop_last=False, shuffle=False, num_workers=torch.cuda.device_count()*4)

    # 测试数据
    #  for i in train_loader:
        #  print(len(i))
        #  print(i[2].shape)
        #  break
    # DONE: 测试igt是从谁到谁的矩阵


    # TODO: trainer 的配置需要增加一些
    #  trainer = pl.Trainer(accelerator='gpu',
            #  devices=[opt.device],
            #  default_root_dir=opt.output_path,
            #  max_steps=opt.epochs)
    checkpoint_callback = ModelCheckpoint(
            dirpath=opt.output_path,
            filename='''{epoch:02d}-loss{train_loss:.8f}''',
            every_n_epochs=opt.save_on_epochs,save_top_k=3,save_on_train_epoch_end=True,
            monitor='train_loss'
            )
    early_stop = early_stopping.EarlyStopping(monitor='epoch_loss', mode='min', stopping_threshold=7.0, check_on_train_epoch_end=True, verbose=opt.verbose, patience=99999)
    trainer = pl.Trainer(accelerator="gpu", devices=[opt.device],
            default_root_dir=opt.output_path,
            max_epochs=opt.epochs,
            callbacks=[checkpoint_callback, early_stop],
            #  log_every_n_steps=3,
            check_val_every_n_epoch=10,
            )
    trainer.fit(model, train_loader, val_dataloaders=tst_loader)
    trainer.test(model, dataloaders=true_testdataset)


if __name__ == '__main__':
    opt = parse_arg()
    per_parse(opt)
    main(opt)

    # 测一下出来的形状的对不对
    #  m=PCTransformer(opt)
    #  a = torch.randn(64,1024,3)
    #  f = m(a)
    #  print(f.shape)

