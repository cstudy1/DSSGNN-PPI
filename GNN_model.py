import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn.pool import global_mean_pool
import numpy as np
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GAT_Top(nn.Module):
    def __init__(self):
        super(GAT_Top, self).__init__()

        self.gat1 = GATConv(384, 48, heads=8, concat=True, dropout=0.3)

        self.gat2 = GATConv(256, 256, dropout=0.3)

        self.gat3 = GATConv(256, 256, dropout=0.3)

        # self.gat4 = GATConv(256, 256, dropout=0.4)

        self.fc1 = nn.Linear(384, 384)

        self.fc5 = nn.Linear(384, 256)

        self.fc2 = nn.Linear(256, 256)

        # self.fc3 = nn.Linear(256, 256)

        self.fc4 = nn.Linear(256, 7)

        self.bn1 = nn.BatchNorm1d(384)

        self.bn2 = nn.BatchNorm1d(256)

        self.bn3 = nn.BatchNorm1d(256)

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.gat3.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc5.reset_parameters()

    # 卷积 归一化 激活 残差 然后再激活
    def forward(self, x, edge_index, train_edge_id, p=0.4):
        x_gate = self.fc1(x)
        x_gate = self.gat1(x_gate, edge_index)
        x_gate = self.bn1(x_gate)
        x = x + x_gate
        x = F.relu(x)

        x = self.fc5(x)
        x_gate = self.gat2(x, edge_index)
        x_gate = self.bn2(x_gate)
        x = x + x_gate
        x = F.relu(x)

        x = self.fc2(x)
        # x_gate = self.gat3(x_gate, edge_index)
        # x_gate = F.relu(x_gate)
        # x = x + x_gate

        # x = F.dropout(x, p=p, training=self.training)

        # x_gate = self.fc3(x)
        # x_gate = self.gat4(x_gate, edge_index)
        # x_gate = F.relu(x_gate)
        # x = x + x_gate
        # x = F.dropout(x, p=p, training=self.training)

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        x = torch.mul(x1, x2)
        x = self.fc4(x)

        return x


class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        adj = adj.to(device)
        h = self.W(x)
        batch_size = h.size()[0]
        N = h.size()[1]
        e = torch.einsum('ijl, ikl->ijk', (torch.matmul(h, self.A), h))
        e = e + e.permute((0, 2, 1))
        zero_vec = -9e15 * torch.ones_like(e)
        zero_vec.to(device)
        e.to(device)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention * adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention, h)))

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))
        retval = coeff * x + (1 - coeff) * h_prime
        return retval


class Top(nn.Module):
    def __init__(self, ):
        super(Top, self).__init__()
        self.gate1 = GAT_gate(384, 384)
        self.gate2 = GAT_gate(256, 256)
        self.gate3 = GAT_gate(128, 128)

        self.fc1 = nn.Linear(384, 384)
        self.fc2 = nn.Linear(384, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 7)

        self.bn1 = nn.BatchNorm1d(384)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def reset_parameters(self):
        self.gate1.reset_parameters()
        self.gate2.reset_parameters()
        self.gate3.reset_parameters()
        self.gate4.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    def forward(self, x, edge_index, adj, train_edge_id):
        x = self.fc1(x)
        x = self.gate1(x, edge_index)
        x = x.view(-1, 384)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.view(1, -1, 384)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        x = self.gate2(x, edge_index)
        x = x.view(-1, 256)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(1, -1, 256)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc3(x)
        x = self.gate3(x, edge_index)
        x = x.view(-1, 128)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(1, -1, 128)
        x = F.dropout(x, p=0.3, training=self.training)

        x = x.squeeze()
        node_id = adj[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        x = torch.mul(x1, x2)
        x = self.fc4(x)
        return x

class GAT1(nn.Module):
    def __init__(self, gat_args):
        super(GAT1, self).__init__()
        # under_module_arg = {'hidden_dim': 128, 'feature_dim': 34}
        hidden_dim = gat_args["hidden_dim"]
        self.conv1 = GATConv(hidden_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)

        self.pool1 = SAGPooling(hidden_dim, dropout=0.4)
        self.pool2 = SAGPooling(hidden_dim, dropout=0.4)
        self.pool3 = SAGPooling(hidden_dim, dropout=0.4)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(gat_args["feature_dim"], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.A = nn.Parameter(torch.randn(hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, batch_data):
        x_feature = batch_data.x
        edge_index = batch_data.edge_index
        edge_distance = batch_data.edge_attr
        batch = batch_data.batch
        x = self.fc1(x_feature)
        # end = global_mean_pool(x, batch)

        x_gnn = self.conv1(x, edge_index, edge_attr=edge_distance)
        x_gnn = self.bn1(x_gnn)
        x = x + x_gnn
        x = F.relu(x)

        y = self.pool1(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        x = self.fc2(x)
        x_gnn = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x + x_gnn
        x = F.relu(x)

        y = self.pool2(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        x = self.fc3(x)
        x_gnn = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x + x_gnn

        y = self.pool3(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        # x = F.relu(x)
        return global_mean_pool(x, batch)
        # return x


class GNN_Model(nn.Module):
    def __init__(self, gat_args: dict):
        super(GNN_Model, self).__init__()
        self.protein_gnn = GAT1(gat_args)
        self.ppi_gnn = Top()

    def forward(self, batch_data, ppi_data):
        ppi_adj = ppi_data.edge_index
        train_edge_id = ppi_data.edge_attr

        embs = self.protein_gnn(batch_data)

        with open('./data/shs27k/SHS27K_protein_bert_vector.pkl', 'rb') as file:
            seq = pickle.load(file)

        numpy_tensor = torch.from_numpy(seq)
        embs = embs.to('cpu')
        result = torch.cat((embs, numpy_tensor), dim=1)
        result = result.cuda(non_blocking=True)

        max_index = ppi_adj.max().item() + 1

        adj = torch.zeros((max_index, max_index), dtype=torch.float)

        adj[ppi_adj[0], ppi_adj[1]] = 1

        adj.to(device)
        result = result.unsqueeze(0)
        final = self.ppi_gnn(result, edge_index=adj, adj=ppi_adj, train_edge_id=train_edge_id)

        return final
