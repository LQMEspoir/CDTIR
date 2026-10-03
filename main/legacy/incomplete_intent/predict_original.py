# =============================================================================
# 文件：legacy/incomplete_intent/predict_original.py
# 模块职责：保留的早期/未完成意图预测脚本。用于复现旧版数据加载、模型构建、标签索引和先验读取逻辑，不作为当前统一训练主路径。
# 主要数据流：旧格式数据/先验文件 -> 模型与标签索引 -> 推理/测试 -> 输出旧版结果。
# 阅读建议：先看本文件的公开函数/类，再沿 import 跟踪到 data_utils、modeling、evaluation 等公共模块。
# 维护说明：本版本仅新增解释性注释，不修改原有表达式、控制流、参数默认值或函数调用关系。
# =============================================================================

import torch
import torch.utils.data

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import pickle
import time
import random
import numpy as np
from collections import OrderedDict
import pandas as pd
from tqdm import trange, tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, hamming_loss


import codecs
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import save
from model.model import BiLSTM
from model.model import HLTDGAT

from utils import get_data as gd
from utils.data import TextDataset
from utils import data_helper as dh

from transformers import BertTokenizer

import warnings
warnings.filterwarnings('ignore') 

parser = argparse.ArgumentParser(description='predict.py')
# opts.model_opts(parser)
parser.add_argument('--batch_size', type=int, default=64, required=False)
parser.add_argument('--embedding', type=str, default='./embedding/ernie-3.0-cpt', required=False)
parser.add_argument('--valid_batch_size', type=int, default=64, required=False)
parser.add_argument('--max_seq_length', type=int, default=512, required=False)
parser.add_argument('--seed', type=int, default=42, required=False)
parser.add_argument('--epochs', type=int, default=100, required=False)
parser.add_argument('--patience', type=int, default=10, required=False)
parser.add_argument('--threshold', type=int, default=0.4, required=False)


opt = parser.parse_args()

opt.num_classes_list = [8, 23, 106]
# cuda
use_cuda = torch.cuda.is_available()
opt.use_cuda = use_cuda

tokenizer = BertTokenizer.from_pretrained(opt.embedding)
opt.tokenizer = tokenizer


if use_cuda:
    device = "cuda:0"
    torch.cuda.set_device("cuda:0")
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
opt.device = device


# -----------------------------------------------------------------------------
# 【函数说明】load_data
# - 职责：读取旧版预测脚本所需数据并转换成后续模型测试所使用的结构。
# - 主要参数：无显式业务参数（仅可能包含 self/cls）。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：会输出日志或告警信息。
# -----------------------------------------------------------------------------
def load_data():
    print('loading data...\n')
    test_data_path = 'predict.xlsx'

    test_data = gd.get_data(test_data_path)
    testset = TextDataset(opt.tokenizer, opt.max_seq_length, test_data)
    
    
    if hasattr(opt, 'valid_batch_size'):
        valid_batch_size = opt.valid_batch_size
    else:
        valid_batch_size = opt.batch_size

    testloader = DataLoader(testset, batch_size=valid_batch_size, drop_last=False)

    
    return {'testset':testset,
            'testloader': testloader }

# -----------------------------------------------------------------------------
# 【函数说明】build_model
# - 职责：按旧版实验配置构建/加载模型及其相关组件。
# - 主要参数：无显式业务参数（仅可能包含 self/cls）。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：会输出日志或告警信息。
# -----------------------------------------------------------------------------
def build_model():
    # model
    print('building model...\n')
    embedding_size = 768
    hidden_size = 512
    label_size = 137
    batch_size = opt.batch_size

    model = HLTDGAT(batch_size, opt.max_seq_length, embedding_size, hidden_size, label_size, opt.embedding,opt.num_classes_list, opt.device)
    # outputs = model.forward(torch.tensor(A), lstm_out)
    
    optim = torch.optim.Adam(model.parameters() ,lr=1e-4)

    return model, optim

# -----------------------------------------------------------------------------
# 【函数说明】test_model
# - 职责：在旧版测试集上逐样本运行模型并汇总预测结果。
# - 主要参数：model、model_path、data、A、B。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：模型推理/张量路径需保持 device 与 dtype 一致；会输出日志或告警信息。
# -----------------------------------------------------------------------------
def test_model(model, model_path, data, A, B):
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    if use_cuda:
        model.to(device)

    # 设置为评估模式
    model.eval()

    # 初始化保存预测结果的列表
    all_preds = []
    testloader = data['testloader']

    with torch.no_grad():  # 确保不计算梯度
        for batch in testloader:
            contexts = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(A, B, contexts, attention_mask)
            pred_on_cpu = preds.cpu().detach().numpy()
            all_preds.append(pred_on_cpu)

    # 检查是否有数据被收集
    if len(all_preds) == 0:
        print("No predictions were made; the data loader may be empty.")
        return np.array([])  # 返回一个空的NumPy数组

    # 将所有预测结果合并为一个数组
    all_preds = np.vstack(all_preds)

    return all_preds


# -----------------------------------------------------------------------------
# 【函数说明】make_label_indices
# - 职责：把标签名称映射成稳定整数索引，供分类头、先验或统计逻辑使用。
# - 主要参数：无显式业务参数（仅可能包含 self/cls）。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：可能读写磁盘文件/模型权重，请保证输出目录可写。
# -----------------------------------------------------------------------------
def make_label_indices():
    # READ HIERARCHY FILE AND MAKE (AND WRITE) DICTIONARY LABEL NAME: LABEL ID
    label_ids = dict()
    with open('tourism.taxonomy', 'r', encoding='utf8') as f:
        for line in f.readlines():
            labels = line.strip().split('\t')
            for label in labels:
                if label  != 'Root' and label not in label_ids.keys():
                    label_ids[label] = len(label_ids)
    with open('labelmetadata.json', 'w') as json_f:
        json.dump(label_ids, json_f)
    return label_ids

# -----------------------------------------------------------------------------
# 【函数说明】read_prior
# - 职责：读取旧版全局先验信息并转换为模型可使用的数据结构。
# - 主要参数：label_ids。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：可能读写磁盘文件/模型权重，请保证输出目录可写。
# -----------------------------------------------------------------------------
def read_prior(label_ids):
    # READ PRIOR WHICH IS A DICTIONARY OF {PARENT: {CHILD 1: PRIOR 1, CHILD 2: PRIOR 2, ...}}
    with open('tourism_probs.json', 'r', encoding='utf8') as f:
        priors = json.load(f)
    top_down_prior = np.zeros((len(label_ids), len(label_ids)))
    bottom_up_prior = np.zeros((len(label_ids), len(label_ids)))
    for parent in priors.keys():
        if parent != 'Root':
            children = priors[parent].keys()
            for child in children:
                top_down_prior[label_ids[parent], label_ids[child]] = priors[parent][child]
                bottom_up_prior[label_ids[child], label_ids[parent]] = 1.
    loop = np.eye(len(label_ids), dtype=np.float32)
    mask = top_down_prior + bottom_up_prior + loop
    return mask

# -----------------------------------------------------------------------------
# 【函数说明】read_level_prior
# - 职责：读取分层/级别先验，并按标签层级建立索引。
# - 主要参数：label_ids。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：可能读写磁盘文件/模型权重，请保证输出目录可写。
# -----------------------------------------------------------------------------
def read_level_prior(label_ids):
    # READ PRIOR WHICH IS A DICTIONARY OF {PARENT: {CHILD 1: PRIOR 1, CHILD 2: PRIOR 2, ...}}
    with open('tourism_level_probs.json', 'r', encoding='utf8') as f:
        levelpriors = json.load(f)
    level_prior = np.zeros((len(label_ids), len(label_ids)))
    for parent in levelpriors.keys():
        if parent != 'Root':
            children = levelpriors[parent].keys()
            for child in children:
                level_prior[label_ids[parent], label_ids[child]] = levelpriors[parent][child]
    mask = level_prior
    return mask
if __name__ == '__main__':
    data_dict = load_data()
    label_ids = make_label_indices()
    A = torch.tensor(read_prior(label_ids))
    A = A.to(device)
    B = torch.tensor(read_level_prior(label_ids))
    B = B.to(device)

    model_path = 'best_model.pth'  # 确保这里是正确的模型路径
    
    model, optim = build_model()

    predictions = test_model(model, model_path, data_dict, A, B)
    print("Predictions:", predictions)
