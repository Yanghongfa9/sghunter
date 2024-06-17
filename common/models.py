"""Defines all graph embedding models"""
import numpy as np
import torch

import networkx as nx
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn.inits import reset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score


#SeedGNN
def masked_softmax(src):
    srcmax1 = src - torch.max(src, dim=1, keepdim=True)[0]
    out1 = torch.softmax(srcmax1, dim=1)

    srcmax2 = src - torch.max(src, dim=0, keepdim=True)[0]
    out2 = torch.softmax(srcmax2, dim=0)

    return (out1 + out2) / 2
class SeedGNN(torch.nn.Module):

    def __init__(self, num_layers, hid):
        super(SeedGNN, self).__init__()
        self.hid = hid
        self.num_layers = num_layers

        self.mlp = torch.nn.ModuleList([Seq(
            Lin(1, hid - 1),
            ReLU(),
        )])
        self.readout = torch.nn.ModuleList([Seq(
            Lin(1, 1),
        )])

        for i in range(1, num_layers):
            self.mlp.append(Seq(
                Lin(hid, hid - 1),
                ReLU(),
            ))
            self.readout.append(Seq(
                Lin(hid, 1),
            ))

    def reset_parameters(self):
        for i in range(self.num_layers):
            reset(self.mlp[i])
            reset(self.readout[i])

    def forward(self, G1, G2, seeds, noisy=False):

        Y_total = []
        n1 = G1.shape[0]
        n2 = G2.shape[0]

        Seeds = torch.zeros([n1, n2])
        Seeds[seeds[0], seeds[1]] = 1

        # S = -torch.ones([n1,n2])/n1
        # S[seeds[0],seeds[1]] = 1
        S = Seeds.unsqueeze(-1).to("cuda")
        # S = S.to("cuda")
        for layeri in range(self.num_layers):

            H = torch.einsum("abh,bc->ach", torch.einsum("ij,jkh->ikh", G1, S), G2)
            if layeri < self.num_layers - 1:
                X = self.mlp[layeri](H) / 1000

            Match = self.readout[layeri](H).squeeze(-1)
            Matchnorm = masked_softmax(Match)
            Matchnorm[seeds[0], :] = 0
            Matchnorm[:, seeds[1]] = 0
            Matchnorm[seeds[0], seeds[1]] = 1
            Y_total.append(Matchnorm)
            Matchnorm_cpu = Matchnorm.cuda().cpu()
            Matchn = Matchnorm_cpu.detach().numpy()
            #Matchn = Matchnorm.detach().numpy()
            row, col = linear_sum_assignment(-Matchn)
            NewSeeds = torch.zeros([n1, n2])
            NewSeeds[row, col] = 10

            Z = (Matchnorm_cpu * NewSeeds).unsqueeze(-1)
            Z = Z.to("cuda")
            S = torch.cat([X, Z], dim=2)

        return Y_total[-1], Y_total

    def loss(self, S, y):

        nll = 0
        EPS = 1e-12
        k = 1
        for Si in S:
            val = Si[y[0], y[1]]
            nll += torch.sum(-torch.log(val + EPS))

        return nll

    def acc(self, S, y):

        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        pred = torch.tensor(col)

        correct = sum(pred[y[0]] == y[1])
        return correct
#首先定义了一个新的 SeedGNNBinaryClassifier 类，该类包含了我们之前定义的 SeedGNN 模型作为基础模型，并添加了一个逻辑回归层作为分类器。
# 然后，我们定义了训练函数 train_model 和评估函数 evaluate，用于训练模型并评估模型性能。
# 最后，我们创建了模型实例，并对其进行训练和评估。
class SeedGNNBinaryClassifier(torch.nn.Module):
    def __init__(self, num_layers, hid):
        super(SeedGNNBinaryClassifier, self).__init__()
        self.seed_gnn = SeedGNN(num_layers, hid)  # 使用前面定义的图匹配模型作为基础模型
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1, 1),  # 输入为相似度，输出为二分类的概率
            torch.nn.Sigmoid()  # 将输出映射到 (0, 1) 之间
        )

    def forward(self, G1, G2, seeds,test=False):
        # 获取基础模型的输出
        similarity, _ = self.seed_gnn(G1, G2, seeds)
        # 将相似度输入到分类器中
        # average_similarity = torch.mean(similarity)
        weights = torch.tensor(np.ones((G1.shape[0], G2.shape[0])))
        for G1_index,G2_index in zip(seeds[0],seeds[1]):
            weights[G1_index,G2_index] = 0
        prob = self.classifier(similarity.unsqueeze(-1))
        # 计算整体相似度的平均值
        weights = weights.to("cuda")
        if test:
            return prob.squeeze(-1)

        average_prob = torch.sum(prob.squeeze(-1)*weights) / torch.sum(weights)

        return  average_prob # 返回二分类的概率值

    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for G1, G2, seeds, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(G1, G2, seeds)
                loss = criterion(outputs, labels.float())  # 计算损失
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    def evaluate(self, test_loader):
        self.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for G1, G2, seeds, labels in test_loader:
                outputs = self.forward(G1, G2, seeds)
                preds = (outputs > 0.5).long()  # 将概率值转换为类别
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {acc}")


class SeedGNN_GMWM(torch.nn.Module):

    def __init__(self, num_layers, hid):
        super(SeedGNN_GMWM, self).__init__()
        self.hid = hid
        self.num_layers = num_layers

        self.mlp = torch.nn.ModuleList([Seq(
            Lin(1, hid - 1),
        )])
        self.readout = torch.nn.ModuleList([Seq(
            Lin(1, 1),
        )])
        # self.alpha = torch.nn.Parameter(torch.zeros(num_layers))

        for i in range(1, num_layers):
            self.mlp.append(Seq(
                Lin(hid, hid - 1),
            ))
            self.readout.append(Seq(
                Lin(hid, 1),
            ))

    def reset_parameters(self):
        for i in range(self.num_layers):
            reset(self.mlp[i])
            reset(self.readout[i])

    def forward(self, G1, G2, seeds, noisy=False):

        Y_total = []
        n1 = G1.shape[0]
        n2 = G2.shape[0]

        Seeds = torch.zeros([n1, n2])
        Seeds[seeds[0], seeds[1]] = 1

        S = Seeds.unsqueeze(-1).to("cuda")

        for layeri in range(self.num_layers):

            H = torch.einsum("abh,bc->ach", torch.einsum("ij,jkh->ikh", G1, S), G2)
            if layeri < self.num_layers - 1:
                X = self.mlp[layeri](H) / 1000

            Match = self.readout[layeri](H).squeeze(-1)
            Matchnorm = masked_softmax(Match)
            Matchnorm[seeds[0], :] = 0
            Matchnorm[:, seeds[1]] = 0
            Matchnorm[seeds[0], seeds[1]] = 1
            Y_total.append(Matchnorm)

            # 使用GMWM算法替代匈牙利算法
            G = nx.Graph()
            for i in range(n1):
                for j in range(n2):
                    G.add_edge(i, j, weight=-Matchnorm[i, j].item())
            matching = nx.max_weight_matching(G, maxcardinality=True)
            NewSeeds = torch.zeros([n1, n2])
            for i, j in matching:
                if i < n1 and j < n2:
                    NewSeeds[i, j] = 10

            Z = (Matchnorm * NewSeeds).unsqueeze(-1)

            S = torch.cat([X, Z], dim=2)

        return Y_total[-1], Y_total





"""Defines all graph embedding models"""
from functools import reduce
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn.inits import reset
from scipy.optimize import linear_sum_assignment

from common import utils
from common import feature_preprocess
import pickle as pc

# GNN -> concat -> MLP graph classification baseline
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred  # .argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)


# Order embedder model -- contains a graph embedding model `emb_model`
class OrderEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False
        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred
        #         diff_emb = emb_bs - emb_as
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
                                                 device=emb_as.device), emb_bs - emb_as) ** 2, dim=1)
        return e

    def criterion(self, pred, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0);
        for negative examples, the e term is trained to be at least greater than self.margin.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e_org = torch.sum(torch.max(torch.zeros_like(emb_as,
                                                     device=utils.get_device()), emb_bs - emb_as) ** 2, dim=1)

        e_org[labels == 0] = torch.max(torch.tensor(0.0,
                                                    device=utils.get_device()), self.margin - e_org)[labels == 0]

        return torch.sum(e_org)


class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        if len(feature_preprocess.FEATURE_AUGMENT) > 0:
            # print(dsfjkjkdfjk)
            self.feat_preprocess = feature_preprocess.Preprocess(input_dim)
            input_dim = self.feat_preprocess.dim_out
        else:
            self.feat_preprocess = None

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, 3 * hidden_dim if
        args.conv_type == "PNA" else hidden_dim))

        conv_model = self.build_conv_model(args.conv_type, 1)
        if args.conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()

        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                                                          self.n_layers))

        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "PNA":
                self.convs_sum.append(conv_model(3 * hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(3 * hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(3 * hidden_input_dim, hidden_dim))
            else:
                if args.conv_type == "SAGE_typed":
                    self.convs.append(conv_model(SAGEConv, hidden_input_dim, hidden_dim, n_edge_types=args.n_edge_type))
                elif args.conv_type == "GCN_typed":
                    self.convs.append(
                        conv_model(pyg_nn.GCNConv, hidden_input_dim, hidden_dim, n_edge_types=args.n_edge_type))
                elif args.conv_type == "GIN_typed":
                    self.convs.append(conv_model(GINConv, hidden_input_dim, hidden_dim, n_edge_types=args.n_edge_type))
                else:
                    self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)
        if args.conv_type == "PNA":
            post_input_dim *= 3

        self.pool = args.pool

        if self.pool == 'set':
            self.poolingLayer = pyg_nn.GraphMultisetTransformer(post_input_dim, args.batch_size, hidden_dim)
        else:
            self.post_mp = nn.Sequential(
                nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 256), nn.ReLU(),
                nn.Linear(256, hidden_dim))
        # self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = args.skip
        self.conv_type = args.conv_type

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            # return lambda i, h: pyg_nn.GINConv(nn.Sequential(
            #    nn.Linear(i, h), nn.ReLU()))
            return lambda i, h: GINConv(i, h)
        elif 'typed' in model_type:
            print(model_type)
            return ConvEdgeType

        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, n_inner_layers)
        elif model_type == "PNA":
            return SAGEConv
        else:
            print("unrecognized model type")

    def forward(self, data):
        x, edge_index, batch, edge_type = data.node_feature, data.edge_index, data.batch, data.type_edge
        indsOfSeeds = torch.where(x[:, 0] == 1)[0]
        x = x.to('cuda')
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs_sum) if self.conv_type == "PNA" else
                       len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                            :i + 1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                                   self.convs_mean[i](curr_emb, edge_index),
                                   self.convs_max[i](curr_emb, edge_index)), dim=-1)
                else:
                    if 'typed' in self.conv_type:
                        x = self.convs[i](curr_emb, edge_index, edge_type)
                    else:
                        x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                                   self.convs_mean[i](emb, edge_index),
                                   self.convs_max[i](emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                if 'typed' in self.conv_type:
                    x = self.convs[i](curr_emb, edge_index, edge_type)
                else:
                    x = self.convs[i](x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        # x = pyg_nn.global_mean_pool(x, batch)
        if self.pool == 'only_seed':
            emb = emb[indsOfSeeds]
            emb = self.post_mp(emb)
        elif self.pool == 'add':
            emb = pyg_nn.global_add_pool(emb, batch)
            emb = self.post_mp(emb)
        elif self.pool == 'mean':
            emb = pyg_nn.global_mean_pool(emb, batch)
            emb = self.post_mp(emb)
        elif self.pool == 'set':
            emb = self.poolingLayer(emb, batch, edge_index=edge_index)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class ConvEdgeType(nn.Module):
    def __init__(self, conv_type, in_channels, out_channels, n_edge_types=1):
        super(ConvEdgeType, self).__init__()

        # pyg_nn.GCNConv, SAGEConv, GINConv

        self.modelAs = nn.ModuleList(
            [conv_type(in_channels, out_channels, aggr="add") for _ in range(n_edge_types)]
        )
        self.modelBs = nn.ModuleList(
            [conv_type(in_channels, out_channels, aggr="add", flow='target_to_source') for _ in range(n_edge_types)]
        )

        self.lin = nn.Linear(2 * out_channels * n_edge_types, out_channels)
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None,
                res_n_id=None):

        edge_index, edge_type = pyg_utils.remove_self_loops(edge_index, edge_attr=edge_type)

        for i in range(len(self.modelAs)):

            if edge_type == None:
                edge_index_type = edge_index
            else:
                eids = torch.nonzero(edge_type == i, as_tuple=False).view(-1)
                srcs = edge_index[0][eids]
                dsts = edge_index[1][eids]
                edge_index_type = torch.stack((srcs, dsts), 0)

            x1 = self.modelAs[i](x, edge_index_type)
            x2 = self.modelBs[i](x, edge_index_type)

            if i == 0:
                emb = torch.cat((x1, x2), dim=1)
            else:
                emb = torch.cat((emb, x1, x2), dim=1)

        return self.lin(emb)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add", flow='source_to_target'):
        super(SAGEConv, self).__init__(aggr=aggr, flow=flow)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
                                    out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        # edge_index, edge_weight = add_remaining_self_loops(
        #    edge_index, edge_weight, 1, x.size(self.node_dim))
        try:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)
            return self.propagate(edge_index, size=size, x=x,
                                  edge_weight=edge_weight, res_n_id=res_n_id)
        except:
            print(edge_index)
            pc.dump([edge_index, size, x, edge_weight, res_n_id], open('edge_index.pkl', 'wb'))
            # dfghjk

    def message(self, x_j, edge_weight):
        # return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)

        aggr_out = self.lin_update(aggr_out)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# pytorch geom GINConv + weighted edges
class GINConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(**kwargs)

        self.nn = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
                                                              edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
                                                          edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)






