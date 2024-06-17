import argparse
import csv
import json
import pickle
from collections import defaultdict
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from load_graph_data import test_layers
from subgraph_matching import config

sys.path.append('/home/yhf/sghunter/')
import dill
import networkx as nx
import pandas as pd
import torch
import pickle as pc
sys.path.insert(0,'helper/')
sys.path.append('/home/yhf/darpa2sysdig/')
from backtrack.core.utils.Graph2Json import json2graph,graph2json
from common import data, utils
sys.path.append('/home/yhf/sghunter/helper/create_pos_neg_dict.py')
from helper import create_pos_neg_dict, utilsDarpha
from subgraph_matching.sghunter_train  import graph_to_adj, combine_data, build_seedgnn, plot_roc_curve


def change_node_feature(graph,path_to_uuid,detected_nodes=None):
    node_index_mapping = {}
    detected_nodes_index = None
    count =0
    removed_nodes = []
    for node, data in graph.nodes(data=True):
        if data['data_node'].nodetype == 'network':
            removed_nodes.append(node)
    graph.remove_nodes_from(removed_nodes)
    for node, data in graph.nodes(data=True):
        node_type = data['data_node'].nodetype
        unidname = data['data_node'].unidname
        flag = False
        count+=1
        for path, uuid_list in path_to_uuid.items():
            if path !='' and (path in unidname or unidname in path):
                # 选择匹配的第一个 UID
                chosen_uuid = uuid_list[0]
                # 更新节点的 uid 和 unidname
                data['data_node'].unid = chosen_uuid
                data['data_node'].unidname = path
                node_index_mapping[node] = chosen_uuid
                # 从字典中删除一个键值对
                if len(path_to_uuid[path]) != 1:
                    path_to_uuid[path].remove(chosen_uuid)
                if detected_nodes == unidname:
                    detected_nodes_index = chosen_uuid
                flag = True
                break  # 匹配成功后跳出循环
            elif 'fluxbox' in unidname and 'fluxbox' in path:
                chosen_uuid = uuid_list[0]
                # 更新节点的 uid 和 unidname
                data['data_node'].unid = chosen_uuid
                data['data_node'].unidname = path
                node_index_mapping[node] = chosen_uuid
                # 从字典中删除一个键值对
                if len(path_to_uuid[path]) != 1:
                    path_to_uuid[path].remove(chosen_uuid)
                if detected_nodes == unidname:
                    detected_nodes_index = chosen_uuid
                flag = True
                break  # 匹配成功后跳出循环
        print(count,flag)
    graph = nx.relabel_nodes(graph, node_index_mapping)
    # for node, data in graph.nodes(data=True):
    #     if data['data_node'].node_type == 'network':
    #         graph.remove_node(node)
    return graph,detected_nodes_index
def find_index_nodefeature(graph,uuid_all,detected_nodes_indexes):
    for node in list(graph.nodes._nodes.keys()):
        if node in uuid_all and node not in detected_nodes_indexes:
            return node
# ti huan shuxing
def index_events(graph):
    event_index = {}   # 为每个事件字段分配的唯一标识

    for u, v, data in graph.edges(data=True):
        if 'data_edge' in data:
            event = data['data_edge'].event
            if event and event not in event_index:
                event_index[event] = len(event_index)  # 为新事件字段分配唯一标识

    return event_index
def replace_edge_attributes(nx_graph, event_index):
    graph =nx.MultiDiGraph()  # 创建一个空的图副本
    # 遍历图中的每条边
    for u, v, data in nx_graph.edges(data=True):
        # 检查是否存在 'data_edge' 字段，并且 'event' 字段存在于 event_index 字典中
        event = data['data_edge'].event
        if event in event_index:
            # 添加节点和边到新图中
            if u not in graph:
                graph.add_node(u)
            if v not in graph:
                graph.add_node(v)
                # 检查是否需要添加边
                if (u, v) not in graph.edges():
                    graph.add_edge(u, v, type_edge=event_index[event])
                elif graph[u][v][0]['type_edge'] != event_index[event]:
                    graph.add_edge(u, v, type_edge=event_index[event])

    return graph
def read_provenance_graph_and_convert_to_adjacency_matrix(filename,uuid_all,path_to_uuid,detected_nodes_indexes,path):

    nx_G = json2graph(path,filename)
    detected_nodes_index = find_index_nodefeature(nx_G,uuid_all,detected_nodes_indexes)
    event_index = index_events(nx_G)
    #detected_nodes_index = []
    nx_g_removeedge = replace_edge_attributes(nx_G, event_index)
    #nx_G_changeNode,detected_nodes_index = change_node_feature(nx_G,path_to_uuid,detected_nodes=filename)
    #graph2json(nx_G_changeNode, filename)
    return  nx_g_removeedge,detected_nodes_index
def read_txt_file(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines
def load_hash2graph(data_dir,numberOfNeighK=3):
  # 从 JSON 文件中读取 hash2graph 对象
    with open(f'{data_dir}{numberOfNeighK}hash2graph.json', 'r') as f:
        hash2graph_json = json.load(f)

    # 创建一个空的字典来存储 MultiDiGraph 对象
    hash2graph_multi_digraph = {}

    # 遍历 hash2graph_json 中的每个键和值
    for key, graph_dict in hash2graph_json.items():
        # 创建一个空的字典来存储图形的 MultiDiGraph 对象
        graph_multi_digraph = {}
        # 遍历图形字典中的每个键和值
        for graph_key, graph_data in graph_dict.items():
            # 如果值是字典，并且具有 MultiDiGraph 数据的结构
            if isinstance(graph_data,dict) and 'directed' in graph_data and 'nodes' in graph_data and 'links' in graph_data:
                # 将图形数据转换为 MultiDiGraph 对象并存储
                graph_multi_digraph[graph_key] = nx.node_link_graph(graph_data)

        # 将当前图形的 MultiDiGraph 对象存储到 hash2graph_multi_digraph 字典中
        hash2graph_multi_digraph[key] = graph_multi_digraph

    return hash2graph_multi_digraph

def load_posQueryHashes(data_dir,numberOfNeighK=3):
    with open(f'{data_dir}{numberOfNeighK}posQueryHashes.json', 'r') as f:
        posQueryHashes = json.load(f)
    for key in posQueryHashes.keys():
        for set_key in posQueryHashes[key].keys():
             posQueryHashes[key][set_key] = set(posQueryHashes[key][set_key])
    return posQueryHashes
def load_data(feature='trace_data',numberOfNeighK=3):
    global procCandidatesTest, procCandidatesTrain
    feature_dir = f'/home/yhf/sghunter/node_feature/{feature}/'
    data_dir = f'/home/yhf/sghunter/data/{feature}/'

    def getType(row):
        tp = row['type']
        feature_dir = f'/home/yhf/sghunter/node_feature/trace_data/'
        if tp in abstarct_indexer:
            return abstarct_indexer[tp]
        else:
            return len(abstarct_indexer)

    #             raise NotImplementedError(f"abstract type {tp} is missing in indexer!")
    abstarct_indexer = pc.load(open(f'{feature_dir}abstarct_indexer.pc', 'rb'))
    nodes_data = pd.read_csv(f'{feature_dir}nodes_data.csv')
    nodes_data = nodes_data.drop_duplicates(subset=['uuid'])
    nodes_data['type_index'] = nodes_data.apply(getType, axis=1)
    nodes_data = nodes_data.set_index('uuid')

    abstractType2array = pc.load(open(feature_dir + 'type2array.pc', 'rb'))
    procName2Feature = utilsDarpha.findProcFeature('/home/yhf/sghunter/gtfobins/*.md')
    procAdd = pc.load(open('/home/yhf/sghunter/gtfobins/procAdd.pkl', 'rb'))

    save_file = f'{data_dir}test_neg_dict_{numberOfNeighK}.pc'
    procCandidatesTest = pc.load(open(save_file, 'rb'))
    save_file = f'{data_dir}train_neg_dict_{numberOfNeighK}.pc'
    procCandidatesTrain = pc.load(open(save_file, 'rb'))
    print(save_file)

    typeAbs = {}
    typeAbs['other'] = set(nodes_data.type)
    typeAbs['proc'] = set([tp for tp in typeAbs['other'] if '_Proc' in tp])
    typeAbs['file'] = set([tp for tp in typeAbs['other'] if '_File' in tp])
    typeAbs['other'] = typeAbs['other'].difference(typeAbs['proc'].union(typeAbs['file']))

    hash2graph = pc.load(open(f'{data_dir}{numberOfNeighK}hash2graph.pkl', 'rb'))
    posQueryHashes = dill.load(open(f'{data_dir}{numberOfNeighK}posQueryHashes.pkl', 'rb'))
    posQueryHashStats = dill.load(open(f'{data_dir}{numberOfNeighK}posQueryHashStats.pkl', 'rb'))
    hash2seed = dill.load(open(f'{data_dir}{numberOfNeighK}hash2seed.pkl', 'rb'))

    save_file=f'/home/yhf/sghunter/data/{feature}/test_neg_dict_3.pc'
    procCandidatesTest = pc.load(open(save_file,'rb'))
    save_file=f'/home/yhf/sghunter/data/{feature}/train_neg_dict_3.pc'
    procCandidatesTrain = pc.load(open(save_file,'rb'))

    # # hash2graph = load_hash2graph(data_dir,numberOfNeighK)
    # # 从 JSON 文件中读取 posQueryHashes 对象
    # posQueryHashes = load_posQueryHashes(data_dir,numberOfNeighK)
    #
    # # 从 JSON 文件中读取 posQueryHashStats 对象
    # with open(f'{data_dir}{numberOfNeighK}posQueryHashStats.json', 'r') as f:
    #     posQueryHashStats = json.load(f)
    #
    # # 从 JSON 文件中读取 hash2seed 对象
    # with open(f'{data_dir}{numberOfNeighK}hash2seed.json', 'r') as f:
    #     hash2seed = json.load(f)


    return nodes_data,hash2graph,posQueryHashes,posQueryHashStats,hash2seed,procCandidatesTest,procCandidatesTrain

# 移至 CUDA 设备的函数
def move_to_device(batch,device='cuda'):
    result = []
    for elem in batch:
        if isinstance(elem, list):
            result.append([e.to(device) for e in elem])
        else:
            result.append(elem.to(device))
    return result

def generate_sghunter_dataset():
    global nodes_data,hash2graph,posQueryHashes,posQueryHashStats,hash2seed,procCandidatesTest,procCandidatesTrain
    posQueryHashes = defaultdict(lambda: defaultdict(set))
    posQueryHashStats = defaultdict(lambda: defaultdict(int))
    nodes_data,hash2graph,posQueryHashes,posQueryHashStats,hash2seed,procCandidatesTest,procCandidatesTrain = load_data()
    # index_nodes_data = nodes_data.index.to_list()
    detections = read_txt_file('/home/yhf/sghunter/query_graph/graph/trace/detection_trace.txt')
    path_to_uuid = {}
    uuid_all=[]

    # #抽象节点以及合并重复节点
    # with open('/home/yhf/sghunter/node_feature/trace_data/all_nodes_data.csv', 'r') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         path = row['path']
    #         uuid_node = row['uuid']
    #         uuid_all.append(uuid_node)
    #         if path!='':
    #             if path in path_to_uuid:
    #                 path_to_uuid[path].append(uuid_node)
    #             else:
    #                 path_to_uuid[path] = [uuid_node]
    #定位关键节点
    with open('/home/yhf/sghunter/node_feature/trace_data/nodes_data.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            path = row['path']
            uuid_node = row['uuid']
            uuid_all.append(uuid_node)

    path = '/home/yhf/sghunter/query_graph/graph/trace/node_prograph/'
    train_dataset = {}
    test_dataset = {}
    detected_nodes_indexes= []
    count = 0
    for detection in detections:
        if count< 8:
            count += 1

            graph,detected_nodes_index = read_provenance_graph_and_convert_to_adjacency_matrix(detection,uuid_all,path_to_uuid,detected_nodes_indexes,path)
            detected_nodes_indexes.append(detected_nodes_index)
            print(detection)
            train_dataset.update({detected_nodes_index:graph})

        else:
            flag = False
            graph,detected_nodes_index = read_provenance_graph_and_convert_to_adjacency_matrix(detection,uuid_all,path_to_uuid,detected_nodes_indexes,path)
            detected_nodes_indexes.append(detected_nodes_index)
            test_dataset.update({detected_nodes_index:graph})
            print(detection)
    graph_dataset =[train_dataset,test_dataset,'graph']
    print('data----over')
    data_identifier ='trace_data'
    numberOfNeighK = 3
    node_anchored  = True
    feature_type='type_basic_Proc'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 512
    val_size = 512
    data_source = data.DiskDataSource(graph_dataset, data_identifier,numberOfNeighK,#6,26,
                    node_anchored=node_anchored, feature_type=feature_type)
    loaders = data_source.gen_data_loaders(batch_size,batch_size, train=True)
    theta = 0.06
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        batch_train = data_source.gen_batch(batch_target, batch_neg_target, batch_neg_query, True,nodes_data=nodes_data,
                                            posQueryHashStats=posQueryHashStats,hash2graph=hash2graph,
                                            posQueryHashes=posQueryHashes,hash2seed=hash2seed,
                                            procCandidatesTest=procCandidatesTest,procCandidatesTrain=procCandidatesTrain)
        #batch = [elem.to(utils.get_device()) for elem in batch]
        #batch_train = move_to_device(batch_train)
        pos_a, pos_b, neg_a, neg_b = batch_train
        pos_a_adj = graph_to_adj(pos_a.G)
        pos_b_adj = graph_to_adj(pos_b.G)
        neg_a_adj = graph_to_adj(neg_a.G)
        neg_b_adj = graph_to_adj(neg_b.G)
        train_pts = combine_data(pos_a_adj, pos_b_adj, neg_a_adj, neg_b_adj, theta)
    loaders = data_source.gen_data_loaders(val_size, batch_size,
                                           train=False, use_distributed_sampling=False)
    batch_n = 0
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        batch_test = data_source.gen_batch(batch_target, batch_neg_target, batch_neg_query, True, nodes_data=nodes_data,
                                           posQueryHashStats=posQueryHashStats, hash2graph=hash2graph,
                                           posQueryHashes=posQueryHashes, hash2seed=hash2seed,
                                           procCandidatesTest=procCandidatesTest,procCandidatesTrain=procCandidatesTrain)
        #batch_test = move_to_device(batch)
        pos_a, pos_b, neg_a, neg_b = batch_test
        pos_a_adj = graph_to_adj(pos_a.G)
        pos_b_adj = graph_to_adj(pos_b.G)
        neg_a_adj = graph_to_adj(neg_a.G)
        neg_b_adj = graph_to_adj(neg_b.G)
        test_pts = combine_data(pos_a_adj, pos_b_adj, neg_a_adj, neg_b_adj, theta)
        batch_n += 1
    batch_train = []
    batch_test = []
    return train_pts,test_pts

if __name__ == '__main__':
    train_pts, test_pts = generate_sghunter_dataset()
    device ='cuda'
    args = argparse.Namespace()
    args.lr = 0.0001
    args.hidden_dim = 16
    args.n_layers = 4
    # layers = [3,4,5,6,7]
    args.model_path = '/home/yhf/sghunter/ckpt/SeedGNN-model-trace-trained.pth4'
    args.filename = 'trace'
    # val_acc = []
    # for layer in layers:
    #     args.n_layers = layer
    #     args.model_path = f'/home/yhf/sghunter/ckpt/trace-{layer}.pth'
    #     model = build_seedgnn(args).to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #     criterion = torch.nn.BCELoss()
    #     val_acc.append(test_layers(train_pts, test_pts, args, model, optimizer, criterion))
    #     print(f'layer {layer} is over')
    # print(val_acc)
    model = build_seedgnn(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.BCELoss()
    for epoch in range(1, 1 + 1000):
        model.train()
        training_loss = 0.0
        training_acc = 0.0
        train_samples = 0
        for data in train_pts:

            G1, G2, seeds, labels = move_to_device(data)
            optimizer.zero_grad()
            outputs = model(G1, G2, seeds, False)
            loss = criterion(outputs, labels.double())
            pred = (outputs > 0.5).long()
            training_acc += torch.sum(pred == labels).item()
            train_samples += 1
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        # 在验证集上评估模型性能
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_raw_preds, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for data in test_pts:
                G1, G2, seeds, labels = move_to_device(data)
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
                f"Train Loss: {training_loss / len(train_pts):.4f}, "
                f"Train Acc: {training_acc / train_samples:.4f}, "
                f"Val Loss: {val_loss / len(test_pts):.4f},"
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
        #
        # if epoch == 10:
        #     plot_roc_curve(all_labels, all_raw_preds)

    path = "/home/yhf/sghunter/ckpt/SeedGNN-model-trace-trained.pth"
    torch.save(model.state_dict(), path)
