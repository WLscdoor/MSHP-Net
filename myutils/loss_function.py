import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE, info_nce

def approx_infoNCE_loss(q, k):
    # 计算query和key的相似度得分
    similarity_scores = torch.matmul(q, k.t())  # 矩阵乘法计算相似度得分

    # 计算相似度得分的温度参数
    temperature = 0.07

    # 计算logits
    logits = similarity_scores / temperature

    # 构建labels（假设有N个样本）
    N = q.size(0)
    labels = torch.arange(N).to(logits.device)

    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    
    return loss

# loss = InfoNCE(negative_mode='unpaired') # negative_mode='unpaired' is the default value
# batch_size, num_negative, embedding_size = 32, 48, 128
# query = torch.randn(batch_size, embedding_size)
# positive_key = torch.randn(batch_size, embedding_size)
# negative_keys = torch.randn(num_negative, embedding_size)
# output = loss(query, positive_key, negative_keys)

class norm_infoNCE_loss(nn.Module):
    def __init__(self):
        super(norm_infoNCE_loss,self).__init__()
        self.infoloss = InfoNCE()

    def forward(self, proj):
        loss = 0
        for i in range(len(proj)):
            query = proj[i]
            query = F.normalize(query.reshape(query.shape[0], -1),p=2,dim=1)
            positive_key = proj[len(proj)-i-1]
            positive_key = F.normalize(positive_key.reshape(positive_key.shape[0], -1),p=2,dim=1)
            loss = loss + self.infoloss(query, positive_key)
        loss = loss/len(proj)

        return loss


class unpair_infoNCE_loss(nn.Module):
    def __init__(self):
        super(unpair_infoNCE_loss,self).__init__()
        self.infoloss = InfoNCE(negative_mode='unpaired')

    def forward(self, proj, org_fea):
        loss = 0
        for i in range(len(proj)):
            query = proj[i]
            query = F.normalize(query.reshape(query.shape[0], -1), p=2, dim=1)
            positive_key = proj[len(proj)-i-1]
            positive_key = F.normalize(positive_key.reshape(positive_key.shape[0], -1),p=2,dim=1)
            negative_keys = torch.cat(org_fea, dim=0)
            negative_keys = F.normalize(negative_keys.reshape(negative_keys.shape[0], -1),p=2,dim=1)
            loss = loss + self.infoloss(query, positive_key, negative_keys)
        loss = loss/len(proj)

        return loss


import torch
import torch.nn.functional as F


def region_dice_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    """
    计算 Dice Loss.
    
    Args:
        prediction (torch.Tensor): 预测张量，大小为 (B, 1, H).
        target (torch.Tensor): 目标张量，大小为 (B, 1, H).
        epsilon (float): 避免除零的微小值.
        
    Returns:
        torch.Tensor: Dice Loss.
    """
    # 计算交集
    intersection = (prediction * target).sum()
    
    # 计算 Dice Loss
    dice = 1 - (2. * intersection + epsilon) / (prediction.sum() + target.sum() + epsilon)
    
    return dice

def multi_scale_region_aware_loss(prediction, label, scale):

    loss = 0
    _, _, HRRPlen = prediction[0].size()
    _, _, HRRPorg = label.size()
    partition = HRRPorg//HRRPlen
    label = label[:,:,0::partition]

    for i in range(0, scale):
        loss = loss + region_dice_loss(prediction[i], label)/len(prediction)
        label = F.max_pool1d(label, kernel_size=2, stride=2)

    return loss


class pair_infoNCE_loss(nn.Module):
    def __init__(self):
        super(pair_infoNCE_loss,self).__init__()
        self.infoloss = InfoNCE(negative_mode='paired')

    def forward(self, proj, org_fea):
        loss = 0
        for i in range(len(proj)):
            query = proj[i]
            query = F.normalize(query.reshape(query.shape[0], -1),p=2,dim=1)
            positive_key = proj[len(proj)-i-1]
            positive_key = F.normalize(positive_key.reshape(positive_key.shape[0], -1),p=2,dim=1)
            negative_keys = torch.stack(org_fea, dim=1)
            negative_keys = F.normalize(negative_keys.reshape(negative_keys.shape[0], negative_keys.shape[1], -1),p=2,dim=1)
            loss = loss + self.infoloss(query, positive_key, negative_keys)
        loss = loss/len(proj)

        return loss


dice_loss = region_dice_loss
cal_dice = multi_scale_region_aware_loss


