"""Train the order embedding model"""
import csv
import uuid

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, auc, roc_curve
# Set this flag to True to use hyperparameter optimization
# We use Testtube for hyperparameter tuning
HYPERPARAM_SEARCH = False
HYPERPARAM_SEARCH_N_TRIALS = None   # how many grid search trials to run
                                    #    (set to None for exhaustive search)

import argparse
from itertools import permutations
import pickle
from queue import PriorityQueue
import os
import random
import time

from deepsnap.batch import Batch
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn

from common import data
from common import models
from common import utils
if HYPERPARAM_SEARCH:
    from test_tube import HyperOptArgumentParser
    from subgraph_matching.hyp_search import parse_encoder
else:
    from subgraph_matching.config import parse_encoder
from subgraph_matching.sghunter_test import validation

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch

import pickle as pc
import sys
sys.path.append("/home/yhf/darpa2sysdig/backtrack/core/utils")
from Graph2Json import json2graph
from ProcessNode import ProcessNode
#5
def build_seedgnn(args):


    model = models.SeedGNNBinaryClassifier(args.n_layers, args.hidden_dim)
    if os.path.exists(args.model_path):
        print('Model is loaded !!!!!!!!!')
        model.load_state_dict(torch.load(args.model_path,
            map_location=utils.get_device()))

    return model

#6
def make_data_source(args,isTrain=True):
    
    dirtyGraph=None
    data_source = data.DiskDataSource(args.dataset, args.data_identifier,args.numberOfNeighK,#6,26,
                node_anchored=args.node_anchored, feature_type=args.feature_type)
    
    return data_source


def graph_to_adj(data_source):
    dataset = []
    for data in data_source:
        dataset.append(nx.adjacency_matrix(data,nodelist=sorted(data.nodes())))
    return dataset

#邻接矩阵正则化
def normalize_adj_matrix(adj_matrix):
    """
    将邻接矩阵归一化到 0 到 1 的范围内

    参数：
    - adj_matrix：邻接矩阵

    返回值：
    - normalized_adj_matrix：归一化后的邻接矩阵
    """
    # 找到最小值和最大值
    min_value = np.min(adj_matrix)
    max_value = np.max(adj_matrix)

    # 将值映射到 0 到 1 的范围内
    normalized_adj_matrix = (adj_matrix - min_value) / (max_value - min_value)

    return normalized_adj_matrix
#种子节点选择
def seed_nodes(G1_nodetypes_to_ids, G2_nodetypes_to_ids):
    search_key = '128.55.12.73:143'
    for key, value in G1_nodetypes_to_ids.items():
        if search_key in key:
            print(key, value)
    for key, value in G2_nodetypes_to_ids.items():
        if search_key in key:
            print(key, value)
    seeds_index_G1_n1 = G1_nodetypes_to_ids['128.55.12.110:36356->128.55.12.73:143']
    seeds_index_G2_n1 = G2_nodetypes_to_ids['128.55.12.110:36356->128.55.12.73:143']
    seeds_index_G1_n2 = G1_nodetypes_to_ids['128.55.12.110:36358->128.55.12.73:143']
    seeds_index_G2_n2 = G2_nodetypes_to_ids['128.55.12.110:36358->128.55.12.73:143']
    seeds_index_G1_p1 = G1_nodetypes_to_ids['/usr/lib/thunderbird/thunderbird']
    seeds_index_G2_p1 = G2_nodetypes_to_ids['/usr/lib/thunderbird/thunderbird']
    seeds_index_G1_p2 = G1_nodetypes_to_ids['/home/admin/Downloads/firefox/firefox http://www.nasa.ng/']
    seeds_index_G2_p2 = G2_nodetypes_to_ids['/home/admin/Downloads/firefox/firefox http://www.nasa.ng/']
    seeds_index_G1_p3 = G1_nodetypes_to_ids['/home/admin/Downloads/firefox/firefox']
    seeds_index_G2_p3 = G2_nodetypes_to_ids['/home/admin/Downloads/firefox/firefox']
    seeds = [torch.tensor([seeds_index_G1_n1,seeds_index_G1_n2,seeds_index_G1_p1,seeds_index_G1_p2,seeds_index_G1_p3], dtype=torch.int64),
             torch.tensor([seeds_index_G2_n1,seeds_index_G2_n2,seeds_index_G2_p1,seeds_index_G2_p2,seeds_index_G2_p3], dtype=torch.int64)]
    return seeds
#将邻接矩阵转换成tensor，并用于后续种子图匹配
# s：节点选择概率，用于生成边。
# alpha：生成边的概率。
# theta：选择节点作为种子节点的概率。
# sample：一个布尔值，表示是否对原始图进行采样，默认为 True。
def gen_Seedgraph(adj_matrix, s,alpha, theta, sample=True):

    # 转换成 PyTorch tensor，并将类型转换为 float
    normalized_adj_matrix = normalize_adj_matrix(adj_matrix)
    adj_dense = normalized_adj_matrix.toarray()
    adj = torch.tensor(adj_dense).float()
    N = adj.shape[0]
    n = N
    if sample:
        indices = torch.randperm(N)[:n]
        adj = adj[indices,:][:,indices]
    # 采样节点之间生成边的概率可以根据实际情况进行调整
    sample = (torch.rand(n,n)<s).float()
    sample = torch.triu(sample,1)
    sample = sample + sample.T
    G1 = adj*sample
    #G1_numpy = G1.numpy()
    # 生成第二个图，调整边生成概率
    sample = (torch.rand(n, n) < alpha * s).float()
    sample = torch.triu(sample, 1)
    sample = sample + sample.T
    G2 = adj * sample
    #G2_numpy = G2.numpy()
    #真实节点
    truth = torch.randperm(n)
    G1 = G1[truth,:][:,truth]
    #种子节点
    indices = seed_nodes(adj_matrix,theta)
    seeds = [indices, truth[indices]]

    return (G1,G2,seeds,truth)


#生成训练正向训练集
#G1 > G2
def combine_data(pos_a, pos_b, neg_a, neg_b,theta):
    combined_data = []

    for adj_mat_a, adj_mat_b in zip(pos_a, pos_b):
        n2 = adj_mat_b.shape[0]  # G2 大小

        # 将邻接矩阵转换为张量形式
        G1_tensor = torch.tensor(adj_mat_a.toarray(), dtype=torch.float32)
        G2_tensor = torch.tensor(adj_mat_b.toarray(), dtype=torch.float32)

        # 生成种子节点对应关系
        T = int(n2 * theta)
        seeds_index_G1 = torch.tensor([i for i in range(T)], dtype=torch.int64)
        seeds_index_G2 = torch.tensor([i for i in range(T)], dtype=torch.int64)
        seeds = [seeds_index_G1, seeds_index_G2]

        # 添加到组合列表中
        combined_data.append((G1_tensor, G2_tensor, seeds,torch.tensor(1)))

    for adj_mat_a, adj_mat_b in zip(neg_a, neg_b):
        n1 = adj_mat_a.shape[0]  # G1 大小
        n2 = adj_mat_b.shape[0]  # G2 大小

        # 将邻接矩阵转换为张量形式
        G1_tensor = torch.tensor(adj_mat_a.toarray(), dtype=torch.float32)
        G2_tensor = torch.tensor(adj_mat_b.toarray(), dtype=torch.float32)

        # 生成随机种子节点对应关系
        seeds_index_G1 = torch.randint(0, n1, (n2,), dtype=torch.int64)
        seeds = [seeds_index_G1, torch.tensor([i for i in range(n2)], dtype=torch.int64)]

        # 添加到组合列表中
        combined_data.append((G1_tensor, G2_tensor, seeds,torch.tensor(0)))

    return combined_data
def plot_roc_curve(labels, raw_pred):
    """
    绘制 ROC 曲线并计算 AUC 值。

    参数：
    - labels: 真实标签，二分类问题中的真实类别。
    - raw_pred: 模型的原始预测概率值，表示样本属于正类的概率。

    返回值：
    - roc_auc: ROC 曲线下方的面积（AUC 值）。
    """
    # 计算 ROC 曲线上的点
    fpr, tpr, thresholds = roc_curve(labels, raw_pred)

    # 计算 AUC 值
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
# def print_graph(adj):
#     # 假设 pos_a 中的第一个图存储在 first_graph_pos_a 中
#     first_graph_pos_a = adj
#
#     # 创建一个 NetworkX 图对象
#     G = nx.from_numpy_array(first_graph_pos_a)
#
#     # 使用 Matplotlib 绘制图形
#     plt.figure(figsize=(8, 6))
#     nx.draw(G, with_labels=True, node_color='skyblue', node_size=800, font_size=12)
#     plt.title("First Graph from pos_a")
#     plt.show()
# def gen_traindataset(train_dataset):
#     datasets = []
#     # graph_para 中的第一个元组 (100, 0.1, 0.6, 0.1) 表示生成一个包含 100 个节点的图形，边的概率为 0.1，选择节点作为种子节点的概率为 0.6，节点之间边存在的概率为 0.1
#     graph_para = [(100, 0.1, 0.6, 0.2), (100, 0.1, 0.8, 0.2), (100, 0.1, 1, 0.2),
#                   (100, 0.3, 0.6, 0.2), (100, 0.3, 0.8, 0.2), (100, 0.3, 1, 0.2),
#                   (100, 0.5, 0.6, 0.2), (100, 0.5, 0.8, 0.2), (100, 0.5, 1, 0.2),
#                   (100, 0.1, 0.6, 0.2)]
#     numgraphs = 10
#     for n, p, s, theta in graph_para:
#         for _ in range(numgraphs):
#             GraphPair = gen_Seedgraph(train_dataset,s,1,theta,sample=True)
#             datasets.append(GraphPair)
#     return datasets
# #4
#读取查询图
def read_query_graph_and_convert_to_adjacency_matrix(file_path):
    # 读取图形文件
    G = nx.read_gexf(file_path, node_type=None)
    G_adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
    G_tensor = torch.tensor(G_adj.toarray(), dtype=torch.float32)
    return G_tensor



#更改uid
def change_node_feature(graph,feature_file):
    path_to_uuid = {}
    #uuid_all = []
    with open(feature_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            path = row['path']
            uuid_node = row['uuid']
            #uuid_all.append(uuid_node)
            if path in path_to_uuid:
                path_to_uuid[path].append(uuid_node)
            else:
                path_to_uuid[path] = [uuid_node]
    node_index_mapping = {}
    network_node_path = []
    for node, data in graph.nodes(data=True):
        node_type = data['data_node'].nodetype
        # if node_type in ['network']:
        #     network_node_path.append(data['data_node'].unidname)
        # if node_type in ['process', 'file']:
        unidname = data['data_node'].unidname
        for path, uuid_list in path_to_uuid.items():
            if path in unidname:
                # 选择匹配的第一个 UID
                chosen_uuid = uuid_list[0]
                # 更新节点的 uid 和 unidname
                data['data_node'].unid = chosen_uuid
                data['data_node'].unidname = path
                node_index_mapping[node] = chosen_uuid
                # 从字典中删除一个键值对
                path_to_uuid[path].remove(chosen_uuid)
                break  # 匹配成功后跳出循环
    graph = nx.relabel_nodes(graph, node_index_mapping)
    return graph
    # for network_node in network_node_path:
    #     new_uid = str(uuid.uuid4())
    #     while new_uid in uuid_all:
    #         new_uid = str(uuid.uuid4())
    #     with open(feature_file, 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow([new_uid, network_node,'network'])

#抽象进程节点
def merge_process_nodes(graph):
    # 创建一个字典，键是pidname，值是与该pidname相关联的所有节点列表
    process_nodes_dict = {}
    for node in graph.nodes(data=True):
        node_id, node_data = node
        if isinstance(node_data['data_node'], ProcessNode):
            pidname = node_data['data_node'].pidname
            if pidname in process_nodes_dict:
                process_nodes_dict[pidname].append(node_id)
            else:
                process_nodes_dict[pidname] = [node_id]

    # 遍历字典中的每个键值对，合并对应的节点并创建新的合并节点
    for pidname, nodes in process_nodes_dict.items():
        merged_node_id = f'merged_{pidname}'
        merged_node_data = {'data_node': ProcessNode('', '', pidname)}
        graph.add_node(merged_node_id, **merged_node_data)

        # 更新图中的边，使它们指向新的合并节点
        for node_id in nodes:
            # 处理后继节点
            for successor_id in graph.successors(node_id):
                if not graph.has_edge(merged_node_id, successor_id):
                    edge_data = graph.get_edge_data(node_id, successor_id)
                    if edge_data:
                        graph.add_edge(merged_node_id, successor_id, attr_dict=edge_data)
            # 处理前驱节点
            for predecessor_id in graph.predecessors(node_id):
                if not graph.has_edge(predecessor_id, merged_node_id):
                    edge_data = graph.get_edge_data(predecessor_id, node_id)
                    if edge_data:
                        graph.add_edge(predecessor_id, merged_node_id, attr_dict= edge_data)

        graph.remove_nodes_from(nodes)

    return graph

#读取溯源图
def read_provenance_graph_and_convert_to_adjacency_matrix(filename):
    nx_G2 = json2graph(filename)
    nx_G2_changeNode = change_node_feature(nx_G2,'/home/yhf/sghunter/node_feature/ta1-theia-e3-official-6r/all_nodes_data.csv')
    nx_G2_merge = merge_process_nodes(nx_G2)
    nodes = nx_G2.nodes()
    nodetypes_to_ids={}
    for i, node in enumerate(nodes):
        if nodes[node]['data_node'].nodetype == 'process':
            print(nodes[node]['data_node'].pidname)
            nodetypes_to_ids.update({nodes[node]['data_node'].pidname: i})
        else:
            nodetypes_to_ids.update({nodes[node]['data_node'].unidname: i})
    G2_adj = nx.adjacency_matrix(nx_G2, nodelist=sorted(nx_G2.nodes()))
    G2_tensor = torch.tensor(G2_adj.toarray(), dtype=torch.float32)
    return G2_tensor,nodetypes_to_ids
def train_loop(args):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")


    print("Using dataset {}".format(args.dataset))

    record_keys = ["conv_type", "n_layers", "hidden_dim",
        "margin", "dataset", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    logger = SummaryWriter(comment=args_str)

    model = build_seedgnn(args)
    #print(model)

    print('create test points')

    test_pts = []
    if args.test:
        #G2,G2_nodetypes_to_ids = read_provenance_graph_and_convert_to_adjacency_matrix('theia_query')
        G1,G1_nodetypes_to_ids = read_provenance_graph_and_convert_to_adjacency_matrix('128.55.12.110:60674->7.149.198.40:80')
        G2,G2_nodetypes_to_ids = read_provenance_graph_and_convert_to_adjacency_matrix('ta1-theia-e3-official-6rphishing')
        # 生成随机种子节点对应关系
        # n1 = G1.shape[0]
        # n2 = G2.shape[0]
        #seeds = seed_nodes(G1_nodetypes_to_ids, G2_nodetypes_to_ids)
        seeds = [torch.tensor([range(0,1000)]),torch.tensor([range(0,1000)])]
        #seeds_index_G1 = torch.randint(0, n1, (n2,), dtype=torch.int64)
        #seeds = [seeds_index_G1, torch.tensor([i for i in range(n2)], dtype=torch.int64)]
        # seeds= [torch.tensor([1205,545]),torch.tensor([33,10])]
        #empty_seeds = torch.zeros(G1.shape[0], G2.shape[0])
        threshold = 0.5
        #seeds =
        # 将 G1 和 G2 作为输入传递给模型，获取输出
        with torch.no_grad():
            outputs = model(G1, G2, seeds,False)

        #row_indices = np.where(outputs > 0.5)[0]
        # 根据模型的输出，判断 G1 和 G2 是否匹配
        if outputs > threshold:
            print("G1 和 G2 匹配")
        else:
            print("G1 和 G2 不匹配")
        return outputs
    
    print('test points are created')
    print('load only train data')

    data_source = make_data_source(args)
    loaders = data_source.gen_data_loaders(args.batch_size, args.batch_size, train=True)
    theta = 0.2
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        batch = data_source.gen_batch(batch_target, batch_neg_target, batch_neg_query, True)
        batch = [elem.to(utils.get_device()) for elem in batch]
        pos_a, pos_b, neg_a, neg_b = batch
        pos_a_adj = graph_to_adj(pos_a.G)
        pos_b_adj = graph_to_adj(pos_b.G)
        neg_a_adj = graph_to_adj(neg_a.G)
        neg_b_adj = graph_to_adj(neg_b.G)
        train_pts = combine_data(pos_a_adj,pos_b_adj,neg_a_adj,neg_b_adj,theta)
    loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
        train=False, use_distributed_sampling=False)
    batch_n = 0
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, False)
        pos_a_adj = graph_to_adj(pos_a.G)
        pos_b_adj = graph_to_adj(pos_b.G)
        neg_a_adj = graph_to_adj(neg_a.G)
        neg_b_adj = graph_to_adj(neg_b.G)
        test_pts = combine_data(pos_a_adj, pos_b_adj, neg_a_adj, neg_b_adj, theta)
        batch_n += 1
    # model.reset_parameters()
    #处理数据
    # train_dataset ,test_dataset = graph_to_adj(data_source)
    graph_train_dataset = train_pts
    graph_test_dataset = test_pts
    # s = 0.8
    # theta = 0.2
    # alpha = 1
    # for adj_matrix in train_dataset:
    #     graph_train_dataset.append(gen_Seedgraph(adj_matrix,s,alpha,theta,sample=True))
    # # graph_train_dataset = gen_traindataset(train_dataset[404])
    # for adj_matrix in test_dataset:
    #     graph_test_dataset.append(gen_Seedgraph(adj_matrix,s,alpha,theta,sample=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=1)
    criterion = torch.nn.BCELoss()

    for epoch in range(1, 1 + 50):
        model.train()
        training_loss = 0.0
        training_acc =0.0
        train_samples =0
        for data in graph_train_dataset:
            G1 = data[0]
            G2 = data[1]
            seeds = data[2]
            # truth = data[3]
            labels = data[3]

            optimizer.zero_grad()
            outputs = model(G1, G2, seeds,False)
            loss = criterion(outputs, labels.double())
            pred =  (outputs > 0.5).long()
            training_acc += torch.sum(pred == labels).item()
            train_samples +=1
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        # acc = (preds == labels).sum()
        # training_acc = acc.item() / len(acc)
        # 在验证集上评估模型性能
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_raw_preds, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for data in graph_test_dataset:
                G1 = data[0]
                G2 = data[1]
                seeds = data[2]
                # truth = data[3]
                labels = data[3]

                outputs = model(G1, G2, seeds)
                loss = criterion(outputs, labels.double())
                val_loss += loss.item()

                preds = (outputs > 0.5).long()
                total_correct += torch.sum(preds == labels).item()
                total_samples += 1
                # 保存预测结果和真实标签，用于后续计算指标
                all_raw_preds.append(outputs.detach().cpu().numpy())
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
        # 打印每个 epoch 的训练损失和验证损失，以及验证集的准确率
        # loss = train(args,model,graph_train_dataset, opt)
        # scheduler.step()
        #
        # 计算指标
        all_raw_preds = np.array(all_raw_preds)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = total_correct / total_samples  # 计算准确率
        auroc = roc_auc_score(all_labels, all_raw_preds)  # 计算AUROC
        avg_prec = average_precision_score(all_labels, all_raw_preds)  # 计算平均精度
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()  # 计算混淆矩阵

        # 计算精确率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0


        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d}, "
                f"Train Loss: {training_loss / len(graph_train_dataset):.4f}, "
                f"Train Acc: {training_acc / train_samples:.4f}, "
                f"Val Loss: {val_loss / len(graph_test_dataset):.4f},"
                f" Val Accuracy: {total_correct / total_samples:.4f}")
            # 输出结果
            print("Validation Loss: {:.4f}".format(val_loss / total_samples))
            print("Accuracy: {:.4f}".format(acc))
            print("Precision: {:.4f}".format(precision))
            print("Recall: {:.4f}".format(recall))
            print("AUROC: {:.4f}".format(auroc))
            print("Average Precision: {:.4f}".format(avg_prec))
            print("Confusion Matrix:")
            print("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))
        #     train_acc = test(args,model,graph_train_dataset)
        #     test_acc = test(args,model,graph_test_dataset)
        #     print(f'epoch {epoch:03d}: Loss: {loss:.8f}, Training Acc: {train_acc:.4f}, Testing Acc: {test_acc:.4f}')
        if epoch == 10:
            plot_roc_curve(all_labels, all_raw_preds)
    # accs = 100 * (test(args, model, graph_test_dataset).numpy())
 
    # print('Acc: ', accs)
    path = "./ckpt/SeedGNN-model-ta1-theia-e3-official-6r-trained.pth"
    torch.save(model.state_dict(), path)

def main(force_test=False):
    parser = (argparse.ArgumentParser(description='Order embedding arguments')
        if not HYPERPARAM_SEARCH else
        HyperOptArgumentParser(strategy='grid_search'))

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True

    if HYPERPARAM_SEARCH:
        for i, hparam_trial in enumerate(args.trials(HYPERPARAM_SEARCH_N_TRIALS)):
            print("Running hyperparameter search trial", i)
            print(hparam_trial)
            train_loop(hparam_trial)
    else:
        train_loop(args)

if __name__ == '__main__':
    main()
