#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: naraysa & akshitac8


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

random.seed(3483)
np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)

def tensordot(x,y):
    return torch.einsum("abc,cd->abd", (x, y)) #矩阵乘法运算,参与运算的两个tensor维度可以不一样

def matmul(x,y):
    return torch.einsum("ab,bc->ac", (x, y)) #是tensor的矩阵乘法运算,参与运算的两个tensor维度 数据类型必须保持一致

class CONV3_3(nn.Module):
    def __init__(self, num_in=512,num_out=512,kernel=3):
        super(CONV3_3, self).__init__()
        self.body = nn.Conv2d(num_in, num_out, kernel, padding=int((kernel-1)/2), dilation=1)
        self.bn = nn.BatchNorm2d(num_out, affine=True, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.body(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class RBF(nn.Module): #基于区域的attention feature(region-based feature)
    """
    Region contextualized block
    """
    def __init__(self, heads=8, d_model=512, d_ff=1024, dropout = 0.1):
        super(RBF, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.w_q = nn.Conv2d(in_channels = d_model , out_channels = d_model , kernel_size=1, bias=True)
        self.w_k = nn.Conv2d(in_channels = d_model , out_channels = d_model , kernel_size=1, bias=True)
        self.w_v = nn.Conv2d(in_channels = d_model, out_channels = d_model, kernel_size=1, bias=True)
        self.w_o = nn.Conv2d(in_channels = d_model , out_channels = d_model , kernel_size=1, bias=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.sub_network = C_R(d_model, d_ff)

    def F_R(self, q, k, v, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        scores = scores.masked_fill(scores == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores) 
        return scores

    def forward(self, q_feat, k_feat, v_feat):
        if k_feat is None:
            k_feat = q_feat
        bs = q_feat.size(0)
        spa = q_feat.size(-1)
        residual = q_feat
        # print("k_feat:",k_feat.shape)
        # print("q_feat:",q_feat.shape)
        # print("v_feat:",v_feat.shape)
        k_h_r = self.w_k(k_feat).view(bs, self.h, self.d_k, spa*spa).transpose(3,2)
        q_h_r = self.w_q(q_feat).view(bs, self.h, self.d_k, spa*spa).transpose(3,2)
        v_h_r = self.w_v(v_feat).view(bs, self.h, self.d_k, spa*spa).transpose(3,2)
        r_h = self.F_R(q_h_r, k_h_r, v_h_r, self.d_k, self.dropout_1)
        alpha_h = torch.matmul(r_h, v_h_r)
        # print("alpha_h:",alpha_h.shape)
        o_r = alpha_h.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        o_r = o_r.permute(0,2,1)
        o_r = o_r.view(-1,self.d_model,spa,spa)
        o_r = self.dropout_2(self.w_o(o_r))
        # print("o_r.shape:", o_r.shape)
        # print("o_r:",o_r)
        # o_r += residual
        # input_o_r = o_r
        # e_r = self.sub_network(o_r)
        # e_r += input_o_r
        e_r = o_r
        return e_r

class C_R(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = d_model , out_channels = d_ff , kernel_size= 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels = d_ff , out_channels = d_model , kernel_size= 1, bias=True)
    def forward(self, x):
        x_out = self.conv2(F.relu(self.conv1(x), True))
        return x_out

class SBF(nn.Module): #基于属性的attention feature(attribute-based feature)
    """
    scene contextualized block
    """
    def __init__(self, opt):
        super(SBF, self).__init__()
        self.channel_dim = opt.channel_dim
        self.sigmoid = nn.Sigmoid()
        self.gcdropout = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.2, False)
        self.w_g = nn.Parameter(nn.init.normal_(torch.empty(300, self.channel_dim)), requires_grad=True)
        # self.gcff = CONV3_3(num_in=self.channel_dim, num_out=self.channel_dim)
        # self.channel_conv = CONV1_1(num_in=self.channel_dim, num_out=self.channel_dim)

    def F_G(self, q , k):
        r_g = q * k
        r_g = self.sigmoid(r_g)
        r_g = r_g.view(-1,self.channel_dim,1)
        return r_g

    def forward(self, h_r, vecs):
        # print("h_r:",h_r.shape)
        h_r = h_r.reshape(-1, 512, 196)
        # print("h_r:",h_r.shape)
        # print("vecs:",vecs.shape)
        A = torch.einsum('iv,vf,bfr->bir',vecs,self.w_g,h_r)
        A = F.softmax(A, dim=-1)
        e_g = torch.einsum('bir,bfr->bif',A, h_r)
        return e_g

class DEML(nn.Module):
    def __init__(self, opt, dim_w2v, dim_feature, w1, w2):
        super(DEML, self).__init__()
        self.w1 = w1
        self.w2 = w2
        # print("vecs:", vecs.shape[0])
        D = dim_feature[1]     #### D is the feature dimension of attention windows
        self.channel_dim = opt.channel_dim
        self.conv_3X3 = CONV3_3(num_out=self.channel_dim)
        self.region_context_block = RBF(heads=opt.heads, d_model=self.channel_dim, d_ff=self.channel_dim*2, dropout = 0.1)
        self.scene_context_block = SBF(opt)
        # self.l2_regularization = torch.tensor(0)

        self.W = nn.Linear(dim_w2v,D, bias=True)
        # self.conv_1X1 = CONV1_1(num_in=self.channel_dim*2, num_out=D)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.g_v = G_V()

    def predict_RBF(self, e_r, vecs, W):
        classifiers = W(vecs) #(925,300)
        m = tensordot(e_r, classifiers.t())
        # print("m_RBF",m)
        # logits_RBF = torch.topk(m, k=1, dim=1)[0]
        # print("m:",m.shape)
        logits_RBF = torch.topk(m,k=6,dim=1)[0].mean(dim=1)
        # print("logits_RBF:", logits_RBF.shape)
        return logits_RBF

    def predict_SBF(self, e_g, vecs, W):
        classifiers = W(vecs) #classifiers(925,512)
        # classifiers = classifiers.reshape(256, -1, 512)
        # print("e_g:",e_g.shape) #e_g: torch.Size([256, 925, 512])
        # m = torch.einsum('bif,fi->bi',e_g, classifiers.t())

        m = tensordot(e_g, classifiers.t()) #
        # print("m_SBF",m.shape)
        logits_SBF = torch.topk(m,k=6,dim=1)[0].mean(dim=1)
        return logits_SBF

    def forward(self, features, vecs):
        # import pdb;pdb.set_trace()
        # print("features:",features.shape)
        # print("vecs:", vecs.shape)
        x_r = features.view([-1,512,14,14]) #是否分成了14*14个区域?
        h_r = self.conv_3X3(x_r)
        # vecs = self.g_v(vecs)
        e_r = self.region_context_block(h_r,h_r,h_r)
        # print("e_r:",e_r.shape)
        e_r = e_r.reshape(-1, 196, 512)
        # print("e_r_reshape:",e_r.shape)
        # print("vecs_forword:",vecs.shape) # (81,300)
        e_g = self.scene_context_block(h_r, vecs)
        logits_RBF = self.predict_RBF(e_r, vecs, self.W)
        logits_SBF = self.predict_SBF(e_g, vecs, self.W)
        # print("logits_RBF:",logits_RBF)
        # print("logits_SBF:", logits_SBF.shape)
        logits = self.w1 * logits_RBF + self.w2 * logits_SBF
        return logits_RBF, logits_SBF, logits

def ranking_lossT(logitsT, labelsT):
    eps = 1e-8
    subset_idxT = torch.sum(torch.abs(labelsT),dim=0)
    subset_idxT = (subset_idxT>0).nonzero().view(-1).long().cuda()
    sub_labelsT = labelsT[:,subset_idxT]
    sub_logitsT = logitsT[:,subset_idxT]
    positive_tagsT = torch.clamp(sub_labelsT,0.,1.)
    negative_tagsT = torch.clamp(-sub_labelsT,0.,1.)
    maskT = positive_tagsT.unsqueeze(1) * negative_tagsT.unsqueeze(-1)
    pos_score_matT = sub_logitsT * positive_tagsT
    neg_score_matT = sub_logitsT * negative_tagsT
    IW_pos3T = pos_score_matT.unsqueeze(1)
    IW_neg3T = neg_score_matT.unsqueeze(-1)
    OT = 1 + IW_neg3T - IW_pos3T
    O_maskT = maskT * OT
    diffT = torch.clamp(O_maskT, 0)
    violationT = torch.sign(diffT).sum(1).sum(1) 
    diffT = diffT.sum(1).sum(1) 
    lossT =  torch.mean(diffT / (violationT+eps))
    return lossT

def js_div(p_output, q_output):
    """
    Function that measures JS divergence between target and output logits:
    """
    eps = 1e-5
    p_output = p_output.cuda()
    q_output = q_output.cuda()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    # if get_softmax:
    log_mean_output = (((p_output + q_output) / 2) + eps)
    log_mean_output = F.softmax(log_mean_output)
    log_mean_output = log_mean_output.log()
    p_output = F.softmax(p_output)
    q_output = F.softmax(q_output)
    jsd = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2
    # jsd = (KLDivLoss(q_output, p_output) + KLDivLoss(q_output, q_output)) / 2
    # print("jsd:",jsd) #torch.tensor
    return jsd #cuda:0

def distill_loss(p1, p2): #每次输入进来的p1 p2都是128阿
    l_dis = 0
    # print("p1:",p1.shape)
    # print("p2:",p2.shape)
    # l_dis = js_div(p1, p2) + square_distance(p1, p2)
    # print("l_dis_before:",l_dis)
    # l_dis = square_distance(p1, p2)
    l_dis = js_div(p1, p2)
    # print("l_dis_before:",l_dis)
    # l_dia = js_div(p1, p2)
    l_dis = torch.mean(l_dis)
    return  l_dis

# def balance_loss(p1, p2):
#     # l_bal = -torch.log(torch.mean(p1))-torch.log(torch.mean(p2))
#     l_bal = -torch.log(torch.mean(p1 + p2))
#     # print("torch.mean(p1):",torch.mean(p1))
#     # print("torch.mean(p2):",torch.mean(p2))
#     return l_bal
